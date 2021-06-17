import numpy as np
from federatedml.secureprotol.jzf_aciq import ACIQ
from arch.api.utils import log_utils
from federatedml.util import consts
from multiprocessing import cpu_count, Pool
from federatedml.secureprotol.jzf_twocomplement import TwoComplement

N_JOBS = cpu_count()
LOGGER = log_utils.getLogger()


def _static_quantize(value, alpha, r_max, int_bits):
    # first clipping
    value = np.clip(value, -alpha, alpha)

    # then quantize
    sign = np.sign(value)
    unsigned_value = value * sign
    unsigned_value = unsigned_value * (pow(2, int_bits - 1) - 1.0) / r_max
    value = unsigned_value * sign

    # then stochastic round
    size = value.shape
    value = np.floor(value + np.random.random(size)).astype(int) # float to int
    value = value.astype(object) # np.int to int (to avoid being tranferred to float later)

    # finally true value to 2's complement representation
    value = TwoComplement.true_to_two(value, int_bits)

    return value


def _static_quantize_padding(value, alpha, int_bits, num_clients):
    # first clipping
    value = np.clip(value, -alpha, alpha)

    # then quantize
    sign = np.sign(value)
    unsigned_value = value * sign
    unsigned_value = unsigned_value * (pow(2, int_bits - 1) - 1.0) / alpha
    value = unsigned_value * sign

    # then stochastic round
    size = value.shape
    value = np.floor(value + np.random.random(size)).astype(int) # float to int
    value = value.astype(object)  # np.int to int (to avoid being tranferred to float later)

    # finally true value to 2's complement representation
    padding_bits = int(np.ceil(np.log2(num_clients)))
    value = TwoComplement.true_to_two(value, int_bits + padding_bits)

    return value


def _static_quantize_padding_asymmetric(value, alpha, int_bits):
    # first clipping and offset
    value = np.clip(value, -alpha, alpha) + alpha

    # then quantize
    value = value * ((1 << int_bits) - 1) / (2 * alpha)

    # then stochastic round
    size = value.shape
    value = np.floor(value + np.random.random(size)).astype(int) # float to int
    value = value.astype(object)  # np.int to int (to avoid being tranferred to float later)

    return value


def _static_unquantize(value, r_max, int_bits):

    # 2's complement representation to true value
    value = TwoComplement.two_to_true(value, int_bits)

    # then unquantize
    sign = np.sign(value)
    unsigned_value = value * sign
    unsigned_value = unsigned_value * r_max / (pow(2, int_bits - 1) - 1.0)
    value = unsigned_value * sign

    return value


def _static_unquantize_padding(value, alpha, int_bits, num_clients):
    factor = int(np.ceil(np.log2(num_clients)))
    int_bits = factor + int_bits
    alpha *= 2 ** factor

    # 2's complement representation to true value
    value &= ((1 << int_bits) - 1)
    value = TwoComplement.two_to_true(value, int_bits)

    # then unquantize
    sign = np.sign(value)
    unsigned_value = value * sign
    unsigned_value = unsigned_value * alpha / (pow(2, int_bits - 1) - 1.0)
    value = unsigned_value * sign

    return value


def _static_unquantize_padding_asymmetric(value, alpha, int_bits, num_clients):
    alpha *= num_clients

    # then unquantize
    value = value * (2 * alpha) / (((1 << int_bits) - 1) * num_clients) - alpha
    return value


def _static_batching(array, int_bits, element_bits, factor):
    element_bits += factor
    batch_size = int_bits // element_bits

    if len(array) % batch_size == 0:
        pass
    else:
        pad_zero_nums = batch_size - len(array) % batch_size
        pad_zeros = [0] * pad_zero_nums
        array = np.append(array, pad_zeros)

    batch_nums = len(array) // batch_size

    ret = []
    mod = 2 ** element_bits
    for b in range(batch_nums):
        temp = 0
        for i in range(batch_size):
            temp *= mod
            temp += array[i + b * batch_size]

        ret.append(temp)

    return np.array(ret).astype(object)


def _static_batching_padding(array, int_bits, element_bits, factor):
    element_bits += factor * 2
    batch_size = int_bits // element_bits

    if len(array) % batch_size == 0:
        pass
    else:
        pad_zero_nums = batch_size - len(array) % batch_size
        pad_zeros = [0] * pad_zero_nums
        array = np.append(array, pad_zeros)

    batch_nums = len(array) // batch_size

    ret = []
    mod = 2 ** element_bits
    for b in range(batch_nums):
        temp = 0
        for i in range(batch_size):
            temp *= mod
            temp += array[i + b * batch_size]

        ret.append(temp)

    return np.array(ret).astype(object)


def _static_batching_padding_asymmetric(array, int_bits, element_bits, factor):
    element_bits += factor
    batch_size = int_bits // element_bits

    if len(array) % batch_size == 0:
        pass
    else:
        pad_zero_nums = batch_size - len(array) % batch_size
        pad_zeros = [0] * pad_zero_nums
        array = np.append(array, pad_zeros)

    batch_nums = len(array) // batch_size

    ret = []
    mod = 2 ** element_bits
    for b in range(batch_nums):
        temp = 0
        for i in range(batch_size):
            temp *= mod
            temp += array[i + b * batch_size]

        ret.append(temp)

    return np.array(ret).astype(object)


def _static_unbatching(array, int_bits, element_bits, factor):
    true_element_bits = element_bits
    element_bits += factor
    batch_size = int_bits // element_bits

    ret = []
    mask = 2 ** element_bits - 1
    for item in array:
        temp = []
        for i in range(batch_size):
            num = item & mask
            temp.append(num)
            item >>= element_bits

        temp.reverse()
        ret += temp
    ret = np.array(ret)

    mod = 2 ** true_element_bits
    ret = ret % mod
    return ret


def _static_unbatching_padding(array, int_bits, element_bits, factor):
    true_element_bits = element_bits + factor
    element_bits += factor * 2
    batch_size = int_bits // element_bits

    ret = []
    mask = 2 ** element_bits - 1
    for item in array:
        temp = []
        for i in range(batch_size):
            num = item & mask
            temp.append(num)
            item >>= element_bits

        temp.reverse()
        ret += temp
    ret = np.array(ret)

    mod = 2 ** true_element_bits
    ret = ret % mod
    return ret


def _static_unbatching_padding_asymmetric(array, int_bits, element_bits, factor):
    element_bits += factor
    batch_size = int_bits // element_bits

    ret = []
    mask = 2 ** element_bits - 1
    for item in array:
        temp = []
        for i in range(batch_size):
            num = item & mask
            temp.append(num)
            item >>= element_bits

        temp.reverse()
        ret += temp
    ret = np.array(ret)

    return ret


class QuantizingBase(object):

    def __init__(self, int_bits, batch, element_bits, secure):
        self.int_bits = int_bits
        self.num_clients = None
        self.iter = 0
        self.element_bits = None
        self.batch = batch
        self.r_max_list = None
        self.alpha_list = None
        self.shape_list = None
        self.secure = secure
        self.layer_size_list = None
        self.element_bits = element_bits

    def set_iter(self, iter):
        self.iter = iter


class QuantizingArbiter(QuantizingBase):

    def __init__(self, int_bits, to_guest, from_guest,
                 to_host, from_host, batch, element_bits, secure):
        super(QuantizingArbiter, self).__init__(int_bits, batch, element_bits, secure)
        self.to_guest = to_guest
        self.from_guest = from_guest
        self.to_host = to_host
        self.from_host = from_host

    def broadcast_num_clients(self, n):
        self.num_clients = n
        self.scatter(obj=n,
                     host_ids=-1,
                     guest_ids=0,
                     suffix=(self.iter, 'num_clients'))

    def set_layer_size_list(self):
        if self.secure:
            return

        guest_message = self.from_guest.get(idx=0,
                                            suffix=(self.iter, -1))
        self.layer_size_list = guest_message

    def scatter(self, obj, host_ids, guest_ids, suffix):
        self.to_host.remote(obj=obj,
                            role=consts.HOST,
                            idx=host_ids,
                            suffix=suffix)
        self.to_guest.remote(obj=obj,
                             role=consts.GUEST,
                             idx=guest_ids,
                             suffix=suffix)

    def help_quantize(self):
        if self.secure:
            return

        guest_list = self.from_guest.get(idx=0, suffix=(self.iter, 0))
        host_lists = self.from_host.get(idx=-1, suffix=(self.iter, 0))

        guest_list = np.array(guest_list)
        host_lists = np.array(host_lists)

        aciq = ACIQ(self.element_bits)
        alphas = []
        self.r_max_list = []
        for i, size in enumerate(self.layer_size_list):
            temp_array = np.append(guest_list[i], host_lists[:, i])
            min = np.min(temp_array)
            max = np.max(temp_array)
            alpha = aciq.get_alpha_gaus(min, max, size)
            alphas.append(alpha)
            r_max = alpha * self.num_clients
            self.r_max_list.append(r_max)

        self.scatter(obj=alphas,
                     host_ids=-1,
                     guest_ids=0,
                     suffix=(self.iter, 0))


class QuantizingClient(QuantizingBase):

    def __init__(self, int_bits, from_arbiter, to_arbiter,
                 batch, element_bits, padding, secure):
        super(QuantizingClient, self).__init__(int_bits, batch, element_bits, secure)
        self.from_arbiter = from_arbiter
        self.to_arbiter = to_arbiter
        self.padding = padding

        self.expected_mean_for_first_round = 0.0
        self.expected_std_for_first_round = 1.0

        if secure:
            self.past_layer_mean_list = []
            self.past_layer_std_list = []

    def receive_num_clients(self):
        self.num_clients = self.from_arbiter.get(idx=0,
                                                 suffix=(self.iter, 'num_clients'))
        return self.num_clients

    def send_layer_size_list(self, weights): # guest
        result = []
        layer_cnt = 0
        for k in weights.walking_order:
            layer_weights = weights._weights[k]
            result.append(layer_weights.size)
            layer_cnt += 1

        self.layer_size_list = result

        if self.secure:
            for i in range(layer_cnt):
                self.past_layer_mean_list.append(self.expected_mean_for_first_round)
                self.past_layer_std_list.append(self.expected_std_for_first_round)
            return

        LOGGER.info("end encoding")  # do matching outer logger action !
        self.to_arbiter.remote(obj=result,
                               role=consts.ARBITER,
                               idx=0,
                               suffix=(self.iter, -1))
        LOGGER.info("start encoding")  # do matching outer logger action !

    def set_layer_size_list(self, weights): # host
        result = []
        layer_cnt = 0
        for k in weights.walking_order:
            layer_weights = weights._weights[k]
            result.append(layer_weights.size)
            layer_cnt += 1

        self.layer_size_list = result

        for i in range(layer_cnt):
            self.past_layer_mean_list.append(self.expected_mean_for_first_round)
            self.past_layer_std_list.append(self.expected_std_for_first_round)

    def quantize(self, weights):
        # suppose that the input is of type OrderDictWeight
        send_list = []
        for k in weights.walking_order:
            layer_weights = weights._weights[k]
            min = np.min(layer_weights)
            max = np.max(layer_weights)
            send_list.append([min, max])

        if self.secure:
            alpha_list = []
            aciq = ACIQ(self.element_bits)

            for i, size in enumerate(self.layer_size_list):
                std = self.past_layer_std_list[i]
                alpha = aciq.get_alpha_gaus_direct(std)
                # LOGGER.info(f"{i} {std} {alpha}")
                if alpha == 0:  # it is because in the latest global model, sigma = 0 (all are the same)
                    alpha = 0.1  # but local update may not be all the same. Hence still need to clip
                alpha_list.append(alpha)
        else:
            pass  # deprecated
            # LOGGER.info("end encoding")  # do matching outer logger action !
            # self.to_arbiter.remote(obj=send_list,
            #                        role=consts.ARBITER,
            #                        idx=0,
            #                        suffix=(self.iter, 0))
            #
            # alpha_list = self.from_arbiter.get(idx=0, suffix=(self.iter, 0))
            # LOGGER.info("begin encoding")  # do matching outer logger action !

        self.r_max_list = []
        self.alpha_list = []

        if self.batch:
            self.shape_list = []

        layer_cnt = 0
        for k in weights.walking_order:
            # to be compatible with sparsification
            if k == 'zzz':
                alpha = 1.0
            else:
                alpha = alpha_list[layer_cnt]
                # LOGGER.info(f"alpha={alpha}")
                r_max = alpha * self.num_clients
                self.r_max_list.append(r_max)
                self.alpha_list.append(alpha)

            layer_weights = weights._weights[k]
            shape = layer_weights.shape
            weights_flatten = layer_weights.flatten()

            # LOGGER.info(f"before quantize {layer_cnt} {weights_flatten[:2]} {weights_flatten[-2:]} {type(weights_flatten[0])}")
            # LOGGER.info(f"alpha {alpha} {type(alpha)}")
            # #     LOGGER.info(f"{type(weights_flatten[0])}")

            if self.batch:
                self.shape_list.append(shape)
            factor = int(np.ceil(np.log2(self.num_clients)))

            if self.padding:
                if self.batch:
                    elements = _static_quantize_padding_asymmetric(weights_flatten, alpha, self.element_bits)

                    # if layer_cnt == 0:
                    #     LOGGER.info(f"after quantize {elements}")
                    #     LOGGER.info(f"{elements[0]}")
                    # LOGGER.info(f"after quantize {layer_cnt} {elements[:2]} {elements[-2:]}")
                    ret = _static_batching_padding_asymmetric(elements, self.int_bits, self.element_bits, factor)
                    # type of element in ret is usually object (big integer)

                    # if layer_cnt == 0:
                    #     LOGGER.info(f"after batching {ret[0]}")

                    weights._weights[k] = ret
                else:
                    elements = _static_quantize_padding_asymmetric(weights_flatten, alpha, self.element_bits)
                    # type of element in ret is np.int64
                    ret = elements.astype(object)
                    # LOGGER.info(f"after quantize {layer_cnt} {ret[:2]} {ret[-2:]} {type(ret[0])}")
                    weights._weights[k] = ret.reshape(shape)
            else:
                pass
                # if self.batch:
                #     elements = _static_quantize(weights_flatten, alpha, r_max, self.element_bits)
                #     ret = _static_batching(elements, self.int_bits, self.element_bits, factor)
                #     # type of element in ret is usually object (big integer)
                #     weights._weights[k] = ret
                # else:
                #     elements = _static_quantize(weights_flatten, alpha, r_max, self.element_bits)
                #     # type of element in elements is np.int64
                #     ret = elements.astype(object)
                #     weights._weights[k] = ret.reshape(shape)

            layer_cnt += 1

        return weights

    def unquantize(self, weights):
        layer_cnt = 0

        for k in weights.walking_order:
            # LOGGER.info(f"{k}")
            r_max = self.r_max_list[layer_cnt]
            alpha = self.alpha_list[layer_cnt]

            layer_weights = weights._weights[k]
            weights_flatten = layer_weights.flatten()

            # LOGGER.info(f"before unquantize {layer_cnt} {weights_flatten[:2]} {weights_flatten[-2:]} {type(weights_flatten[0])}")

            factor = int(np.ceil(np.log2(self.num_clients)))

            if self.padding:
                if self.batch:
                    shape = self.shape_list[layer_cnt]
                    size = np.prod(shape)
                    # if layer_cnt == 0:
                    #     LOGGER.info(f"before unbatching {weights_flatten[0]}")
                    weights_flatten = _static_unbatching_padding_asymmetric(weights_flatten, self.int_bits,
                                                                 self.element_bits, factor)
                    weights_flatten = weights_flatten[:size]
                    # LOGGER.info(f"before unquantize {layer_cnt} {weights_flatten[:2]} {weights_flatten[-2:]}")
                    ret = _static_unquantize_padding_asymmetric(weights_flatten, alpha, self.element_bits, self.num_clients)
                else:
                    shape = layer_weights.shape
                    ret = _static_unquantize_padding_asymmetric(weights_flatten, alpha, self.element_bits, self.num_clients)
            else:
                pass
                # if self.batch:
                #     shape = self.shape_list[layer_cnt]
                #     size = np.prod(shape)
                #     weights_flatten = _static_unbatching(weights_flatten, self.int_bits,
                #                                          self.element_bits, factor)
                #     weights_flatten = weights_flatten[:size]
                #     ret = _static_unquantize(weights_flatten, r_max, self.element_bits)
                # else:
                #     shape = layer_weights.shape
                #     ret = _static_unquantize(weights_flatten, r_max, self.element_bits)

            weights._weights[k] = ret.reshape(shape)

            # LOGGER.info(f"after unquantize {layer_cnt} {ret[:2]} {ret[-2:]} {type(ret[0])}")
            layer_cnt += 1

        return weights

    def normalize(self, weights):
        layer_cnt = 0
        for k in weights.walking_order:
            weights._weights[k] -= self.past_layer_mean_list[layer_cnt]
            layer_cnt += 1
        return weights

    def unnormalize(self, weights):
        layer_cnt = 0
        for k in weights.walking_order:
            # TODO
            # currently we think that the model to be unquantized
            # should be exactly the new global model
            # while this may not be the case in other settings
            weights._weights[k] += self.past_layer_mean_list[layer_cnt]
            mean = np.mean(weights._weights[k])
            self.past_layer_mean_list[layer_cnt] = mean
            std = np.std(weights._weights[k])
            self.past_layer_std_list[layer_cnt] = std
            # LOGGER.info(f"mean {self.past_layer_mean_list[layer_cnt]}")
            layer_cnt += 1

        return weights
