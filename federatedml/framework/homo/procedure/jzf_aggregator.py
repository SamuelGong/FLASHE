import functools
import types
import typing
import numpy as np
from functools import reduce
from arch.api.utils import log_utils
import time
import copy
from federatedml.framework.homo.sync import loss_transfer_sync, is_converge_sync
from federatedml.framework.jzf_weights import JZFNumericWeights as NumericWeights
from federatedml.framework.jzf_weights import JZFWeights as Weights
from federatedml.framework.jzf_weights import _to_bytes, _from_bytes
from federatedml.framework.jzf_weights import JZFOrderDictWeights as OrderDictWeights
from federatedml.util import consts
from federatedml.framework.homo.procedure import jzf_plain_block as plain_block
from federatedml.framework.homo.procedure import jzf_additive_mask_block as additive_mask_block
from federatedml.framework.homo.procedure import jzf_paillier_block as paillier_block
from federatedml.framework.homo.procedure import jzf_flashe_block as flashe_block
from federatedml.framework.homo.procedure import jzf_simple_block as simple_block
from federatedml.framework.homo.procedure import jzf_bfv_block as bfv_block
from federatedml.framework.homo.procedure import jzf_ckks_block as ckks_block
from federatedml.transfer_variable.base_transfer_variable import Variable

LOGGER = log_utils.getLogger()


def _static_add(x, y):
    if isinstance(x, list):
        ret = []
        for xi, yi in zip(x, y):
            ret.append(_static_add(xi, yi))
        return ret
    else:
        return x + y


def _static_add_2(x, y):
    _w = dict()
    for k in x.walking_order:
        _w[k] = _static_add(y._weights[k], x._weights[k])
    return OrderDictWeights(_w, bits=x._bits,
                            need_compress_for_next_remote=x._need_compress_for_next_remote,
                            size_dict=x._size_dict, shape=x._shape)


def cipher_from_bytes(cipher, l):
    ret = []
    tmp_cnt = 0
    for x in l:
        # LOGGER.info(f"{tmp_cnt}")
        _w = dict()
        for k in x.walking_order:
            _w[k] = cipher.from_bytes(x._weights[k])
        ret.append(OrderDictWeights(_w, bits=x._bits,
                                    need_compress_for_next_remote=x._need_compress_for_next_remote,
                                    size_dict=x._size_dict, shape=x._shape))
        tmp_cnt += 1
    return ret


def cipher_to_bytes(cipher, x):
    _w = dict()
    for k in x.walking_order:
        _w[k] = cipher.to_bytes(x._weights[k])
    return OrderDictWeights(_w, bits=x._bits,
                            need_compress_for_next_remote=x._need_compress_for_next_remote,
                            size_dict=x._size_dict, shape=x._shape)


# partition and merge are exclusive for paillier+no batching
# due to memory/network limitations of FATE
def partition(weights, num_partitions):
    shape_list = []
    d_list = [{} for _ in range(num_partitions)]
    partitioned_weights = []
    bits = weights._bits

    for k in weights.walking_order:
        w = weights._weights[k]
        shape_list.append(w.shape)

        w = w.flatten()
        parts = np.array_split(w, num_partitions)
        for i in range(num_partitions):
            d_list[i][k] = parts[i]

    for d in d_list:
        partitioned_weights.append(OrderDictWeights(d, bits))

    return partitioned_weights, shape_list


def merge_at_arbiter(model_partitions_list):
    shape_list = model_partitions_list[0][0].get_shape_list()
    walking_order = model_partitions_list[0][0].weights.walking_order

    upload_models = []
    for client_idx, partitions in enumerate(model_partitions_list):
        layer_cnt = 0
        for k in walking_order:
            layer_flatten = None
            for part_idx, part in enumerate(partitions):
                if part_idx == 0:
                    layer_flatten = part._weights[k]
                else:
                    layer_flatten = np.append(layer_flatten, part._weights[k])
            layer = layer_flatten.reshape(shape_list[layer_cnt])
            partitions[0]._weights[k] = layer
            layer_cnt += 1

        upload_models.append(partitions[0])

    return upload_models


def merge_at_client(partitions):
    weights = partitions[0].weights
    shape_list = partitions[0].get_shape_list()

    layer_cnt = 0
    for k in weights.walking_order:
        layer_flatten = None
        for part_idx, part in enumerate(partitions):
            if part_idx == 0:
                layer_flatten = part._weights[k]
            else:
                layer_flatten = np.append(layer_flatten, part._weights[k])
        layer = layer_flatten.reshape(shape_list[layer_cnt])
        partitions[0]._weights[k] = layer
        layer_cnt += 1

    return partitions[0]


class Arbiter(object):
    def __init__(self):
        self._loss_sync = None
        self._converge_sync = None
        self.model = None
        self.arbiter_to_guest = None
        self.guest_to_arbiter = None
        self.arbiter_to_host = None
        self.host_to_arbiter = None
        self.sparsity = 1.0
        self.secure_aggregate = None
        self.secure_aggregate_args = None
        self._secure_aggregate_cipher = None
        self.num_hosts = None

    def expand_to_dense(self, models, masks, total):
        client_idx = 0
        for mo, ma in zip(models, masks):
            only_key = list(mo._weights.keys())[0] # assume total flattening
            a = mo._weights[only_key]
            zero = a[-1]
            a = a[:-1]

            expand_model = np.zeros(total, dtype=object)
            expand_model[ma] = a

            zero_location = list(set(np.arange(total).tolist()) - set(ma))
            expand_model[zero_location] = zero
            mo._weights[only_key] = expand_model
            # LOGGER.info(f"expand {client_idx} {expand_model[:5]}")
            client_idx += 1

    def register_aggregator(self, transfer_variables, sparsity=1.0, secure_aggregate="plain",
                            secure_aggregate_args=None):
        self.sparsity = sparsity
        self.secure_aggregate = secure_aggregate
        self.secure_aggregate_args = secure_aggregate_args

        self.num_hosts = len(Variable.roles_to_parties([consts.HOST]))
        # this is a static method
        # you need to pass an array!

        if secure_aggregate == "plain":
            self._secure_aggregate_cipher =  plain_block.Arbiter(
                secure_aggregate_args
            ).register_plain_cipher(transfer_variables).create_quantizer()
        elif secure_aggregate == "additive":
            self._secure_aggregate_cipher = additive_mask_block.Arbiter(
                secure_aggregate_args
            ).register_additive_mask_cipher(transfer_variables).create_cipher()
        elif secure_aggregate == "paillier":
            self._secure_aggregate_cipher = paillier_block.Arbiter(
                secure_aggregate_args
            ).register_paillier_cipher(transfer_variables).create_cipher()
        elif secure_aggregate == "flashe":
            self._secure_aggregate_cipher = flashe_block.Arbiter(
                secure_aggregate_args
            ).register_flashe_cipher(transfer_variables).create_cipher()
        elif secure_aggregate == "simple":
            self._secure_aggregate_cipher = simple_block.Arbiter(
                secure_aggregate_args
            ).register_simple_cipher(transfer_variables).create_cipher()
        elif secure_aggregate == "bfv":
            self._secure_aggregate_cipher = bfv_block.Arbiter(
                secure_aggregate_args
            ).register_bfv_cipher(transfer_variables).create_cipher()
        elif secure_aggregate == "ckks":
            self._secure_aggregate_cipher = ckks_block.Arbiter(
                secure_aggregate_args
            ).register_ckks_cipher(transfer_variables).create_cipher()

        self._loss_sync = loss_transfer_sync.Arbiter().register_loss_transfer(
            host_loss_transfer=transfer_variables.host_loss,
            guest_loss_transfer=transfer_variables.guest_loss)

        self._converge_sync = is_converge_sync.Arbiter().register_is_converge(
            is_converge_variable=transfer_variables.is_converge)

        self.arbiter_to_host = transfer_variables.arbiter_to_host
        self.arbiter_to_guest = transfer_variables.arbiter_to_guest
        self.guest_to_arbiter = transfer_variables.guest_to_arbiter
        self.host_to_arbiter = transfer_variables.host_to_arbiter

        return self

    def aggregate_model(self, iter_index, ciphers_dict=None, suffix=tuple()):
        self._secure_aggregate_cipher.set_iter_index(iter_index)
        self._secure_aggregate_cipher.help_quantize()

        if self.secure_aggregate == "additive":
            self._secure_aggregate_cipher.help_encrypt()

        LOGGER.info("begin collect")
        # one by one for flow control especially dealing with Paillier
        models = []
        degrees = []
        idx_lists = []

        if not self.sparsity == 1.0:  # if sparsification is employed
            tu = self.host_to_arbiter.get(idx=-1, suffix=suffix + ('mask',))
            obj = self.guest_to_arbiter.get(idx=0, suffix=suffix + ('mask',))
            masks = [_from_bytes(t['encoded_masked_locations'], t['l'], t['bits']) for t in tu]
            masks = [_from_bytes(obj['encoded_masked_locations'], obj['l'], obj['bits'])] + masks
            total = obj['total']
            for mask in masks:  # IMPORTANT~
                mask.reverse()
            # temp = _from_bytes(tu[0]['encoded_masked_locations'], tu[0]['l'], tu[0]['bits'])
            # LOGGER.info(f"{tu[0]['encoded_masked_locations']} {tu[0]['l']} {tu[0]['bits']}")
            # LOGGER.info(f"first {masks[0][:5]} {masks[0][-5:]} {len(masks[0])}")
            if self.secure_aggregate in ["flashe"]:
                self._secure_aggregate_cipher.dynamic_masking(masks, total, suffix)


        try_count = 0
        mode = self.secure_aggregate_args["mode"]
        num_partitions = self.secure_aggregate_args["num_partitions"]

        if mode == "parallel":

            if num_partitions > 1:
                model_partitions_list = []
                for part_idx in range(num_partitions):
                    self.arbiter_to_guest.remote(obj="allow_upload", role=consts.GUEST,
                                                 idx=0, suffix=suffix + ('allow_upload', part_idx))
                    self.arbiter_to_host.remote(obj="allow_upload", role=consts.HOST,
                                                idx=-1, suffix=suffix + ('allow_upload', part_idx))

                    self.host_to_arbiter.clean()
                    self.guest_to_arbiter.clean()
                    m = self.guest_to_arbiter.get(idx=0,
                                                  suffix=suffix + ('upload_model', part_idx))
                    ms = self.host_to_arbiter.get(idx=-1,
                                                  suffix=suffix + ('upload_model', part_idx))
                    if part_idx == 0:
                        model_partitions_list.append([m])
                        for m in ms:
                            model_partitions_list.append([m])
                    else:
                        model_partitions_list[0].append(m)
                        for idx, m in enumerate(ms):
                            model_partitions_list[idx + 1].append(m)

                    self.arbiter_to_guest.remote(obj="finish", role=consts.GUEST,
                                                 idx=0, suffix=suffix + ('upload_result', part_idx))
                    self.arbiter_to_host.remote(obj="finish", role=consts.HOST,
                                                idx=-1, suffix=suffix + ('upload_result', part_idx))

                uploaded_models = merge_at_arbiter(model_partitions_list)
                for uploaded_model in uploaded_models:
                    degrees.append(uploaded_model.get_degree())
                    idx_lists.append(uploaded_model.get_idx_list())
                    if not self.sparsity == 1.0:
                        models.append(uploaded_model.get_weights(need_decompress=True))
                    else:
                        models.append(uploaded_model.weights)

            else:
                uploaded_model = self.guest_to_arbiter.get(idx=0,
                                                           suffix=suffix + ('upload_model', try_count))
                degrees.append(uploaded_model.get_degree())
                idx_lists.append(uploaded_model.get_idx_list())
                if not self.sparsity == 1.0:
                    models.append(uploaded_model.get_weights(need_decompress=True))
                else:
                    models.append(uploaded_model.weights)

                uploaded_models = self.host_to_arbiter.get(idx=-1, suffix=suffix + ('upload_model', try_count))
                for uploaded_model in uploaded_models:
                    degrees.append(uploaded_model.get_degree())
                    idx_lists.append(uploaded_model.get_idx_list())
                    if not self.sparsity == 1.0:
                        models.append(uploaded_model.get_weights(need_decompress=True))
                    else:
                        models.append(uploaded_model.weights)

        else:
            self.arbiter_to_guest.remote(obj="allow_upload", role=consts.GUEST,
                                         idx=0, suffix=suffix + ('allow_upload',))

            failed = True
            while failed:
                try:
                    self.guest_to_arbiter.clean()
                    uploaded_model = self.guest_to_arbiter.get(idx=0,
                                                               suffix=suffix + ('upload_model', try_count))
                    failed = False
                except TypeError:
                    LOGGER.info(f"[GET] attempt {try_count} failed due to TypeError")
                    self.arbiter_to_guest.remote(obj="resend", role=consts.GUEST,
                                                 idx=0, suffix=suffix + ('upload_result', try_count))
                    try_count += 1

            degrees.append(uploaded_model.get_degree())
            idx_lists.append(uploaded_model.get_idx_list())
            if not self.sparsity == 1.0:
                models.append(uploaded_model.get_weights(need_decompress=True))
            else:
                models.append(uploaded_model.weights)
            self.arbiter_to_guest.remote(obj="finish", role=consts.GUEST,
                                         idx=0, suffix=suffix + ('upload_result', try_count))

            for host_comm_idx in range(self.num_hosts):
                self.arbiter_to_host.remote(obj="allow_upload", role=consts.HOST,
                                            idx=host_comm_idx, suffix=suffix + ('allow_upload',))
                failed = True
                try_count = 0
                while failed:
                    try:
                        self.host_to_arbiter.clean()
                        uploaded_model = self.host_to_arbiter.get(idx=host_comm_idx,
                                                                  suffix=suffix + ('upload_model', try_count))
                        failed = False
                    except TypeError:
                        LOGGER.info(f"[GET] attempt {try_count} failed due to TypeError")
                        self.arbiter_to_host.remote(obj="resend", role=consts.HOST,
                                                    idx=host_comm_idx, suffix=suffix + ('upload result', try_count))
                        try_count += 1

                degrees.append(uploaded_model.get_degree())
                idx_lists.append(uploaded_model.get_idx_list())

                if not self.sparsity == 1.0:
                    models.append(uploaded_model.get_weights(need_decompress=True))
                else:
                    models.append(uploaded_model.weights)
                self.arbiter_to_host.remote(obj="finish", role=consts.HOST,
                                            idx=host_comm_idx, suffix=suffix + ('upload_result', try_count))
        if not self.secure_aggregate == "bfv":
            models = np.array(models)

        LOGGER.info("end collect")

        size_dict = models[0]._size_dict
        if size_dict is None:
            is_compressed = False
        else:
            is_compressed = True
            # LOGGER.info(f"{size_dict}")
            # since we have flattened the entire model
            # we actually only have one key here
            size = 0
            for k in size_dict.keys():
                size += size_dict[k]

        LOGGER.info("begin aggregate")
        total_degree = reduce(lambda x, y: x + y, degrees)

        if not self.sparsity == 1.0:
            self.expand_to_dense(models, masks, total)
            is_compressed = False

        if self.secure_aggregate in ["plain"]:
            if 'quantize' in self.secure_aggregate_args:
                int_bits = self._secure_aggregate_cipher.int_bits
                if is_compressed:
                    # total_model = copy.deepcopy(models[0])
                    # for k in total_model.walking_order:
                    #     mod = 1 << (size_dict[k] * int_bits)
                    #     total_model._weight[k] = reduce(
                    #         lambda x, y: (x._weights[k] + y._weights[k]) % mod, models)

                    # since we have flattened the entire model
                    # mod = 1 << (int_bits * size)
                    total_model = reduce(lambda x, y: x + y, models)
                else:
                    # mod = 1 << int_bits
                    total_model = reduce(lambda x, y: x + y, models)
            else:
                total_model = reduce(lambda x, y: x + y, models)
        elif self.secure_aggregate in ["flashe"]:
            int_bits = self._secure_aggregate_cipher.int_bits
            if is_compressed:
                # total_model = copy.deepcopy(models[0])
                # for k in total_model.walking_order:
                #     mod = 1 << (size_dict[k] * int_bits)
                #     total_model._weight[k] = reduce(
                #         lambda x, y: (x._weights[k] + y._weights[k]) % mod, models)

                # since we have flattened the entire model
                mod = 1 << (int_bits * size)
                # for p, model in enumerate(models):
                #     for k in model.walking_order:
                #         LOGGER.info(f"{p}: {model._weights[k]}")
                #         break
                total_model = reduce(lambda x, y: (x + y) % mod, models)
                # for k in total_model.walking_order:
                #     LOGGER.info(f"total: {total_model._weights[k]}")
                #     break
            else:
                mod = 1 << int_bits
                # for p, model in enumerate(models):
                    # for k in model.walking_order:
                    #     LOGGER.info(f"{p}: {model._weights[k][:2]}")
                    #     break

                total_model = reduce(lambda x, y: (x + y) % mod, models)
                # for k in total_model.walking_order:
                #     LOGGER.info(f"total: {total_model._weights[k][:2]}")
                #     break

            # LOGGER.info(f"before dispatch {total_model._weights[total_model.walking_order[0]][:5]}")
            # LOGGER.info(f"{type(total_model._weights[total_model.walking_order[0]][0])}")
        elif self.secure_aggregate in ["paillier"]:
            mod = self._secure_aggregate_cipher.get_n() ** 2
            total_model = reduce(lambda x, y: (x * y) % mod, models)  # remember it is "*" not "+" for Paillier
        elif self.secure_aggregate in ["bfv"]:
            # total_model = self._secure_aggregate_cipher.cipher.sum(models)
            models = cipher_from_bytes(self._secure_aggregate_cipher.cipher, models)
            total_model = reduce(lambda x, y: _static_add_2(x, y), models)
            total_model = cipher_to_bytes(self._secure_aggregate_cipher.cipher, total_model)
        elif self.secure_aggregate in ["ckks"]:
            models = cipher_from_bytes(self._secure_aggregate_cipher.cipher, models)
            total_model = reduce(lambda x, y: _static_add_2(x, y), models)
            total_model = cipher_to_bytes(self._secure_aggregate_cipher.cipher, total_model)
        else:
            total_model = reduce(lambda x, y: x + y, models)

        if self.secure_aggregate in ["simple", "additive", "flashe"]:
            total_idx_list = reduce(lambda x, y: x + y, idx_lists)
        else:
            total_idx_list = None
        LOGGER.debug("In aggregate model, total_degree: {}".format(total_degree))
        LOGGER.info("end aggregate")

        if self.secure_aggregate == "additive":
            LOGGER.info("begin decryption")
            self._secure_aggregate_cipher.prepare_decrypt(total_idx_list)
            total_model.decrypted(cipher=self._secure_aggregate_cipher, inplace=True)
            LOGGER.info("end decryption")
            LOGGER.info("begin decoding")
            total_model = self._secure_aggregate_cipher.unquantize(total_model)
            LOGGER.info("end decoding")

        return total_model, total_degree, total_idx_list

    def aggregate_and_broadcast(self, iter_index, ciphers_dict=None, suffix=tuple()):
        model, degrees, idx_lists = self.aggregate_model(ciphers_dict=ciphers_dict,
                                                         iter_index=iter_index, suffix=suffix)

        try_count = 0
        mode = self.secure_aggregate_args["mode"]
        num_partitions = self.secure_aggregate_args["num_partitions"]
        if mode == "parallel":
            if num_partitions > 1:
                partitioned_weights, shape_list = partition(model, num_partitions)
                for part_idx in range(num_partitions):
                    remote_partition = partitioned_weights[part_idx]
                    remote_partition = remote_partition.for_remote()
                    remote_partition.with_degree(degrees)
                    remote_partition.with_idx_list(idx_lists)
                    remote_partition.with_shape_list(shape_list)

                    LOGGER.info("begin dispatch")
                    self.arbiter_to_guest.remote(remote_partition, role=consts.GUEST, idx=0,
                                                 suffix=suffix + ('agg_model', part_idx))
                    self.arbiter_to_host.remote(remote_partition, role=consts.HOST, idx=-1,
                                                suffix=suffix + ('agg_model', part_idx))
                    LOGGER.info("end dispatch")

                    dispatch_result = self.guest_to_arbiter.get(idx=0, suffix=suffix + ("dispatch_result", part_idx))
                    dispatch_results = self.host_to_arbiter.get(idx=-1, suffix=suffix + ("dispatch_result", part_idx))
                    dispatch_results = [dispatch_result] + dispatch_results
                    LOGGER.info(f"dispatch_result: {dispatch_results}")

                    self.arbiter_to_guest.clean()
                    self.arbiter_to_host.clean()
            else:
                model = model.for_remote()
                model.with_degree(degrees)
                model.with_idx_list(idx_lists)
                LOGGER.info("begin dispatch")
                self.arbiter_to_guest.remote(model, role=consts.GUEST, idx=0, suffix=suffix + ('agg_model', try_count))
                self.arbiter_to_host.remote(model, role=consts.HOST, idx=-1,
                                            suffix=suffix + ('agg_model', try_count))
                LOGGER.info("end dispatch")
        else:
            model = model.for_remote()
            model.with_degree(degrees)
            model.with_idx_list(idx_lists)
            # TO DO: with dropout things will not be so simple
            for host_comm_idx in range(self.num_hosts):
                try_count = 0
                while True:
                    LOGGER.info("begin dispatch")
                    if try_count > 0:
                        self.arbiter_to_host.clean()
                    self.arbiter_to_host.remote(model, role=consts.HOST, idx=host_comm_idx,
                                                suffix=suffix + ('agg_model', try_count))
                    LOGGER.info("end dispatch")
                    dispatch_result = self.host_to_arbiter.get(idx=host_comm_idx,
                                                               suffix=suffix + ("dispatch_result", try_count))
                    LOGGER.info(f"dispatch_result: {dispatch_result}")
                    if dispatch_result == "finish":
                        break
                    try_count += 1
                self.arbiter_to_host.clean()

            try_count = 0
            while True:
                LOGGER.info("begin dispatch")
                if try_count > 0:
                    self.arbiter_to_guest.clean()
                self.arbiter_to_guest.remote(model, role=consts.GUEST, idx=0, suffix=suffix + ('agg_model', try_count))
                LOGGER.info("end dispatch")
                dispatch_result = self.guest_to_arbiter.get(idx=0, suffix=suffix + ("dispatch_result", try_count))
                LOGGER.info(f"dispatch_result: {dispatch_result}")
                if dispatch_result == "finish":
                    break
                try_count += 1
            self.arbiter_to_guest.clean()

    def aggregate_loss(self, idx=None, suffix=tuple()):
        losses = self._loss_sync.get_losses(idx=idx, suffix=suffix)
        total_loss = 0.0
        total_degree = 0.0
        for loss in losses:
            total_loss += loss.unboxed
            total_degree += loss.get_degree(1.0)
        return total_loss / total_degree

    def send_converge_status(self, converge_func: types.FunctionType, converge_args, suffix=tuple()):
        return self._converge_sync.check_converge_status(converge_func=converge_func, converge_args=converge_args,
                                                         suffix=suffix)


class Client(object):
    def __init__(self):
        self._secure_aggregate_cipher = None
        self._model_scatter = None
        self._loss_sync = None
        self._converge_sync = None
        self._sparsity = 1.0
        self.remain_weights = None
        self.shape_dict_used_for_sparsification = None
        self.weights_last_round = None
        self._secure_aggregate = "plain"
        self._secure_aggregate_args = None
        self.to_arbiter = None
        self.from_arbiter = None
        self.shape_list = None
        self.shape_dict = None
        self.degree = None

    def sparsify(self, weights):
        if self.remain_weights is None:
            self.remain_weights = {}

        base = 0
        locations = []
        shapes = {}
        # layer-wise top s% sparsification
        for k in weights.walking_order:
            layer = weights._weights[k]
            shape = layer.shape
            shapes[k] = shape
            size = np.prod(shape)

            flatten = layer.flatten()
            abs_flatten = np.abs(flatten)
            if k in self.remain_weights:
                flatten += self.remain_weights[k]

            idx = max(1, int(np.floor(self._sparsity * size)))
            location = sorted(abs_flatten.argsort()[-idx:][::-1])
            masked_layer = flatten[location] # a compacted one
            weights._weights[k] = masked_layer

            flatten[location] = 0.0
            self.remain_weights[k] = flatten

            locations += (np.array(location) + base).astype(object).tolist()
            # LOGGER.info(f"{len(location)} {size}")
            base += size

        if self.shape_dict_used_for_sparsification is None:
            self.shape_dict_used_for_sparsification = shapes

        # LOGGER.info(f"mask {locations[:5]}")
        LOGGER.info(f"number of masked weights: {len(locations)}")
        # LOGGER.info(f"{locations}")
        # since base is of np.int64 and does not have bit_length()
        base = base.item()
        bits = base.bit_length()
        encoded_locations, le = _to_bytes(locations, bits)
        loc = _from_bytes(encoded_locations, le, bits)
        loc.reverse()
        # LOGGER.info('=======')
        # LOGGER.info(f"{loc}")
        return encoded_locations, le, bits, base

    def flatten_weights(self, weights):
        new_weights = []
        new_weights = np.array(new_weights)
        shape_dict = {}

        first_k = None
        for k in weights.walking_order:
            if first_k is None:
                first_k = k
            layer_weights = weights._weights[k]

            if k == "zzz":
                pass
            else:
                shape = layer_weights.shape
                shape_dict[k] = shape
            weights_flatten = layer_weights.flatten()
            new_weights = np.append(new_weights, weights_flatten)
            del weights._weights[k]

        self.shape_dict = shape_dict
        weights._weights[first_k] = new_weights

        # you need to refresh walking order!
        weights.walking_order = sorted(weights._weights.keys(), key=str)
        return weights

    def unflatten_weights(self, weights):

        # should only contain one key
        only_key = None
        for k in weights.walking_order:
            only_key = k
            break

        flatten_weights = weights._weights[only_key]
        for k, shape in self.shape_dict.items():
            # LOGGER.info(f"{k} {shape}")
            size = np.prod(shape)
            layer_flatten_weights = flatten_weights[:size]
            flatten_weights = flatten_weights[size:]
            layer_weights = layer_flatten_weights.reshape(shape)
            weights._weights[k] = layer_weights

        # you need to refresh walking order!
        weights.walking_order = sorted(weights._weights.keys(), key=str)
        return weights

    def secure_aggregate(self, weights: Weights, before: Weights = None, iter_index: int = 0, suffix=tuple(),
                         degree: float = None, secure_aggregate="plain", is_model=False):
        if degree:
            weights *= degree
            LOGGER.info(f"degree: {degree}")
            self.degree = degree

        if is_model:
            if secure_aggregate in ["plain"]:
                if 'quantize' in self._secure_aggregate_args:
                    weight_bits = self._secure_aggregate_args['quantize']['int_bits']
                else:
                    weight_bits = None
                # weight_bits = None
            elif secure_aggregate in ["flashe"]:
                weight_bits = self._secure_aggregate_args['quantize']['int_bits']
                # weight_bits = None
            # elif secure_aggregate in ["paillier"] and self._secure_aggregate_cipher.batch:
            #     n_square = self._secure_aggregate_cipher.get_n() ** 2
            #     weight_bits = n_square.bit_length()
            else:
                weight_bits = None
            weights.set_bits(weight_bits)

            self._secure_aggregate_cipher.set_iter_index(iter_index)

            if not self._sparsity == 1.0:  # if sparsification is employed
                self.weights_last_round = before
                weights = weights - before
                # don't need to return weights as it is an inplace operation
                encoded_masked_locations, l, bits, total = self.sparsify(weights)

                # this is because directly trasmitting a 4-tuple could fail
                if secure_aggregate in ["flashe"]:
                    self._secure_aggregate_cipher.cipher.total = total
                obj = {'encoded_masked_locations': encoded_masked_locations, 'bits': bits, 'l': l, 'total': total}
                self.to_arbiter.remote(obj=obj, role=consts.ARBITER,
                                       idx=0, suffix=suffix + ("mask",))
                if secure_aggregate in ["flashe"]:
                    self._secure_aggregate_cipher.dynamic_masking(suffix)

            weights = self._secure_aggregate_cipher.normalize(weights)
            if not self._sparsity == 1.0:
                weights._weights['zzz'] = np.array([0.0])
                weights.refresh_walking_order()
                # both are combined to to make sure quantized zero
                # will be contained at the end of flattened weight

            LOGGER.info("begin encoding")
            weights = self._secure_aggregate_cipher.quantize(weights)
            weights = self.flatten_weights(weights)
            LOGGER.info("end encoding")

            if secure_aggregate in ["plain"]:
                pass
            elif secure_aggregate in ["simple", "additive", "flashe", "paillier", "bfv", "ckks"]:
                LOGGER.info("begin encryption")
                # LOGGER.info(f"before encryption {weights._weights[weights.walking_order[0]][:5]} "
                #             f"{weights._weights[weights.walking_order[0]][-1]}")

                if secure_aggregate == "additive":
                    self._secure_aggregate_cipher.prepare_encrypt()
                if secure_aggregate in ['flashe'] and (not self._sparsity == 1.0):
                    zero_quantized = weights._weights[weights.walking_order[0]][-1]
                    weights._weights[weights.walking_order[0]] = weights._weights[weights.walking_order[0]][:-1]

                weights = weights.encrypted(cipher=self._secure_aggregate_cipher, inplace=True)

                if secure_aggregate in ['flashe'] and (not self._sparsity == 1.0):
                    weights._weights[weights.walking_order[0]] = np.append(weights._weights[weights.walking_order[0]],
                                                                           [zero_quantized])

                # LOGGER.info(f"after encryption {weights._weights[weights.walking_order[0]][:5]} "
                #             f"{weights._weights[weights.walking_order[0]][-1]}")
                LOGGER.info("end encryption")

            mode = self._secure_aggregate_args["mode"]
            num_partitions = self._secure_aggregate_args["num_partitions"]

            if mode == "parallel" and num_partitions > 1:
                partitioned_weights, shape_list = partition(weights, num_partitions)
                self.shape_list = shape_list

                for part_idx, w in enumerate(partitioned_weights):
                    if part_idx == 0:
                        w = w.for_remote().with_degree(degree) if degree else weights.for_remote()
                        w = w.with_shape_list(shape_list)
                        if secure_aggregate in ["simple", "additive", "flashe"]:
                            idx_list = self._secure_aggregate_cipher.get_idx_list()
                            w = w.with_idx_list(idx_list)
                        partitioned_weights[part_idx] = w
                    else:
                        partitioned_weights[part_idx] = w.for_remote()
            else:
                remote_weights = weights.for_remote().with_degree(degree) if degree else weights.for_remote()
                if secure_aggregate in ["simple", "additive", "flashe"]:
                    idx_list = self._secure_aggregate_cipher.get_idx_list()
                    remote_weights.with_idx_list(idx_list)

            try_count = 0
            if mode == "parallel":
                if num_partitions > 1:
                    for part_idx in range(num_partitions):
                        _ = self.from_arbiter.get(idx=0, suffix=suffix + ("allow_upload", part_idx))
                        LOGGER.info("begin upload")
                        self.to_arbiter.remote(obj=partitioned_weights[part_idx], role=consts.ARBITER,
                                               idx=0, suffix=suffix + ("upload_model", part_idx))
                        LOGGER.info("end upload")
                        upload_result = self.from_arbiter.get(idx=0, suffix=suffix + ("upload_result", part_idx))
                        LOGGER.info(f"upload_result for partition {part_idx}: {upload_result}")
                        self.to_arbiter.clean()
                else:
                    self.to_arbiter.remote(obj=remote_weights, role=consts.ARBITER,
                                           idx=0, suffix=suffix + ("upload_model", try_count))
            else:
                _ = self.from_arbiter.get(idx=0, suffix=suffix + ("allow_upload",))
                while True:
                    LOGGER.info("begin upload")
                    if try_count > 0:
                        self.to_arbiter.clean()
                    self.to_arbiter.remote(obj=remote_weights, role=consts.ARBITER,
                                           idx=0, suffix=suffix + ("upload_model", try_count))
                    LOGGER.info("end upload")
                    upload_result = self.from_arbiter.get(idx=0, suffix=suffix + ("upload_result", try_count))
                    LOGGER.info(f"upload_result: {upload_result}")
                    if upload_result == 'finish':
                        break
                    try_count += 1
                self.to_arbiter.clean()
        else:  # remote loss
            loss = weights
            remote_loss = loss.for_remote().with_degree(degree) if degree else loss.for_remote()
            self._loss_sync.send_loss(remote_loss, suffix)

        return

    def send_model(self, weights: Weights, before: Weights, iter_index: int,
                   degree: float = None, suffix=tuple()):
        return self.secure_aggregate(weights=weights, degree=degree, iter_index=iter_index, suffix=suffix,
                                     secure_aggregate=self._secure_aggregate, is_model=True, before=before)

    def aggregate_then_get(self, model: Weights, iter_index: int, before: Weights = None,
                           degree: float = None, suffix=tuple()) -> Weights:
        self.send_model(weights=model, degree=degree,
                        suffix=suffix, iter_index=iter_index,
                        before=before)

        if self._secure_aggregate in ["flashe"]:
            LOGGER.info("begin prepare_decrypt")
            self._secure_aggregate_cipher.prepare_decrypt()
            LOGGER.info("end prepare_decrypt")
            LOGGER.info("begin prepare_encrypt")
            self._secure_aggregate_cipher.prepare_encrypt()
            LOGGER.info("end prepare_encrypt")

        return self.get_aggregated_model(suffix=suffix)

    def get_aggregated_model(self, suffix=tuple()):
        if self._secure_aggregate in ["additive"]:
            self._secure_aggregate_cipher.help_decrypt()

        LOGGER.info("begin download")
        failed = True
        try_count = 0

        mode = self._secure_aggregate_args["mode"]
        num_partitions = self._secure_aggregate_args["num_partitions"]
        if mode == "parallel":
            if num_partitions > 1:

                aggregated_model_partitions = []

                for part_idx in range(num_partitions):
                    partition = self.from_arbiter.get(idx=0, suffix=suffix + ('agg_model', part_idx))
                    aggregated_model_partitions.append(partition)
                    self.to_arbiter.remote(obj="finish", role=consts.ARBITER, idx=0,
                                           suffix=suffix + ("dispatch_result", part_idx))
                    self.from_arbiter.clean()

                aggregated_model = merge_at_client(aggregated_model_partitions)

            else:
                self.from_arbiter.clean()
                aggregated_model = self.from_arbiter.get(idx=0, suffix=suffix + ('agg_model', try_count))
                self.to_arbiter.remote(obj="finish", role=consts.ARBITER, idx=0,
                                       suffix=suffix + ("dispatch_result", try_count))
        else:
            while failed:
                try:
                    self.from_arbiter.clean()
                    aggregated_model = self.from_arbiter.get(idx=0, suffix=suffix + ('agg_model', try_count))
                    failed = False
                except TypeError:
                    LOGGER.info(f"[GET] attempt {try_count} failed due to TypeError")
                    self.to_arbiter.remote(obj="resend", role=consts.ARBITER,
                                           idx=0, suffix=suffix + ('dispatch_result', try_count))
                    try_count += 1
            self.to_arbiter.remote(obj="finish", role=consts.ARBITER,
                                   idx=0, suffix=suffix + ("dispatch_result", try_count))

        degrees = aggregated_model.get_degree()
        total_idx_list = aggregated_model.get_idx_list()
        LOGGER.info("end download")

        weight = aggregated_model.get_weights(need_decompress=True)
        del aggregated_model

        if self._secure_aggregate in ["paillier", "simple", "flashe", "bfv", "ckks"]:
            LOGGER.info("begin decryption")
            if self._secure_aggregate in ["flashe", "simple"]:
                self._secure_aggregate_cipher.set_idx_list(total_idx_list)

            # LOGGER.info(f"before decryption {weight._weights[weight.walking_order[0]][:5]}")
            # LOGGER.info(f"{type(weight._weights[weight.walking_order[0]][0])}")
            weight = weight.decrypted(cipher=self._secure_aggregate_cipher, inplace=True)
            LOGGER.info("end decryption")

        LOGGER.info("begin decoding")
        # LOGGER.info(f"before unflatten {weight._weights[weight.walking_order[0]][:5]}")
        # LOGGER.info(f"{type(weight._weights[weight.walking_order[0]][0])}")
        # LOGGER.info(f"before unflatten {weight.walking_order}")
        if not self._sparsity == 1.0:
            self.shape_dict = self.shape_dict_used_for_sparsification
        weight = self.unflatten_weights(weight)

        # LOGGER.info(f"after unflatten {weight.walking_order}")
        weight = self._secure_aggregate_cipher.unquantize(weight)
        LOGGER.info("end decoding")

        weight /= (degrees / self.degree)
        LOGGER.info(f"total degree: {degrees}")
        weight = self._secure_aggregate_cipher.unnormalize(weight)
        weight /= self.degree

        weight = weight + self.weights_last_round

        return weight

    def send_loss(self, loss: typing.Union[float, Weights], degree: float = None, suffix=tuple()):
        if isinstance(loss, float):
            loss = NumericWeights(loss)
        return self.secure_aggregate(weights=loss, degree=degree, suffix=suffix,
                                     secure_aggregate="plain", is_model=False)

    def get_converge_status(self, suffix=tuple()):
        return self._converge_sync.get_converge_status(suffix=suffix)


class Guest(Client):
    def __init__(self):
        super(Guest, self).__init__()

    def register_aggregator(self, transfer_variables, sparsity=1.0, secure_aggregate="plain", secure_aggregate_args=None):
        if secure_aggregate_args is None:
            secure_aggregate_args = [None]

        self._sparsity = sparsity
        self._secure_aggregate = secure_aggregate
        self._secure_aggregate_args = secure_aggregate_args
        if secure_aggregate == "plain":
            self._secure_aggregate_cipher = plain_block.Guest(
                secure_aggregate_args
            ).register_plain_cipher(
                transfer_variables
            ).create_quantizer()
        elif secure_aggregate == "additive":
            self._secure_aggregate_cipher = additive_mask_block.Guest(
                secure_aggregate_args
            ).register_additive_mask_cipher(
                transfer_variables).create_cipher()
        elif secure_aggregate == "paillier":
            self._secure_aggregate_cipher = paillier_block.Guest(
                secure_aggregate_args
            ).register_paillier_cipher(
                transfer_variables).create_cipher()
        elif secure_aggregate == "flashe":
            self._secure_aggregate_cipher = flashe_block.Guest(
                secure_aggregate_args
            ).register_flashe_cipher(
                transfer_variables).create_cipher()
        elif secure_aggregate == "simple":
            self._secure_aggregate_cipher = simple_block.Guest(
                secure_aggregate_args
            ).register_simple_cipher(
                transfer_variables).create_cipher()
        elif secure_aggregate == "bfv":
            self._secure_aggregate_cipher = bfv_block.Guest(
                secure_aggregate_args
            ).register_bfv_cipher(
                transfer_variables).create_cipher()
        elif secure_aggregate == "ckks":
            self._secure_aggregate_cipher = ckks_block.Guest(
                secure_aggregate_args
            ).register_ckks_cipher(
                transfer_variables).create_cipher()

        self._loss_sync = loss_transfer_sync.Guest().register_loss_transfer(
            loss_transfer=transfer_variables.guest_loss)

        self._converge_sync = is_converge_sync.Guest().register_is_converge(
            is_converge_variable=transfer_variables.is_converge)

        self.from_arbiter = transfer_variables.arbiter_to_guest
        self.to_arbiter = transfer_variables.guest_to_arbiter

        return self


class Host(Client):
    def __init__(self):
        super(Host, self).__init__()

    def register_aggregator(self, transfer_variables, sparsity=1.0, secure_aggregate="plain",
                            secure_aggregate_args=None):
        self._sparsity = sparsity
        self._secure_aggregate = secure_aggregate
        self._secure_aggregate_args = secure_aggregate_args
        if secure_aggregate == "plain":
            self._secure_aggregate_cipher = plain_block.Host(
                secure_aggregate_args
            ).register_plain_cipher(
                transfer_variables
            ).create_quantizer()
        elif secure_aggregate == "additive":
            self._secure_aggregate_cipher = additive_mask_block.Host(
                secure_aggregate_args).register_additive_mask_cipher(
                transfer_variables).create_cipher()
        elif secure_aggregate == "paillier":
            self._secure_aggregate_cipher = paillier_block.Host(
                secure_aggregate_args
            ).register_paillier_cipher(
                transfer_variables).create_cipher()
        elif secure_aggregate == "flashe":
            self._secure_aggregate_cipher = flashe_block.Host(
                secure_aggregate_args
            ).register_flashe_cipher(
                transfer_variables).create_cipher()
        elif secure_aggregate == "simple":
            self._secure_aggregate_cipher = simple_block.Host(
                secure_aggregate_args
            ).register_simple_cipher(
                transfer_variables).create_cipher()
        elif secure_aggregate == "bfv":
            self._secure_aggregate_cipher = bfv_block.Host(
                secure_aggregate_args
            ).register_bfv_cipher(
                transfer_variables).create_cipher()
        elif secure_aggregate == "ckks":
            self._secure_aggregate_cipher = ckks_block.Host(
                secure_aggregate_args
            ).register_ckks_cipher(
                transfer_variables).create_cipher()

        self._loss_sync = loss_transfer_sync.Host().register_loss_transfer(
            loss_transfer=transfer_variables.host_loss)

        self._converge_sync = is_converge_sync.Host().register_is_converge(
            is_converge_variable=transfer_variables.is_converge)

        self.from_arbiter = transfer_variables.arbiter_to_host
        self.to_arbiter = transfer_variables.host_to_arbiter

        return self


def with_role(role, transfer_variable, sparsity=1.0, secure_aggregate="plain", secure_aggregate_args=None):
    if role == consts.GUEST:
        return Guest().register_aggregator(transfer_variable, sparsity, secure_aggregate, secure_aggregate_args)
    elif role == consts.HOST:
        return Host().register_aggregator(transfer_variable, sparsity, secure_aggregate, secure_aggregate_args)
    elif role == consts.ARBITER:
        return Arbiter().register_aggregator(transfer_variable, sparsity, secure_aggregate, secure_aggregate_args)
    else:
        raise ValueError(f"role {role} not found")
