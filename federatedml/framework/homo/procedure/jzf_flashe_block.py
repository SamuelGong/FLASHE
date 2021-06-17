from arch.api.utils import log_utils
from federatedml.framework.homo.sync import identify_uuid_sync, jzf_dh_keys_exchange_sync
from federatedml.secureprotol.jzf_flashe import FlasheCipher
from federatedml.secureprotol.jzf_aes import AESCipher
from federatedml.util import consts
from federatedml.secureprotol.jzf_quantize import QuantizingClient, QuantizingArbiter
import numpy as np

LOGGER = log_utils.getLogger()

class Arbiter(identify_uuid_sync.Arbiter,
              jzf_dh_keys_exchange_sync.Arbiter):

    def __init__(self, args):
        super(Arbiter, self).__init__()
        self.arbiter_to_guest = None
        self.arbiter_to_host = None
        self.guest_to_arbiter = None
        self.host_to_arbiter = None
        self.guest_uuid = None

        self.int_bits = args['quantize']['int_bits']
        self.batch = args['quantize']['batch']
        self.element_bits = args['quantize']['element_bits']
        self.secure = args['quantize']['secure']
        if 'mask' in args:
            self.mask = args['mask']
        else:
            self.mask = 'double'

        self.quantizer = None
        self.has_set_layer_size_list = False

    def set_iter_index(self, iter_index):
        self.quantizer.set_iter(iter_index)

    def register_flashe_cipher(self, transfer_variables):
        self.arbiter_to_guest = transfer_variables.arbiter_to_guest
        self.arbiter_to_host = transfer_variables.arbiter_to_host
        self.guest_to_arbiter = transfer_variables.guest_to_arbiter
        self.host_to_arbiter = transfer_variables.host_to_arbiter

        self.register_identify_uuid(guest_uuid_trv=transfer_variables.guest_uuid,
                                    host_uuid_trv=transfer_variables.host_uuid,
                                    conflict_flag_trv=transfer_variables.uuid_conflict_flag)

        self.register_dh_key_exchange(dh_pubkey_trv=transfer_variables.dh_pubkey,
                                      dh_ciphertext_guest_trv=transfer_variables.dh_ciphertext_guest,
                                      dh_ciphertext_host_trv=transfer_variables.dh_ciphertext_host,
                                      dh_ciphertext_bc_trv=transfer_variables.dh_ciphertext_bc)
        return self

    def create_cipher(self):
        LOGGER.info("synchronizing uuid")
        self.validate_uuid()

        LOGGER.info("Diffie-Hellman keys exchanging")
        self.key_exchange()

        LOGGER.info("Forwarding encrypted meta prng seeds for Guest to Hosts")
        shared_dict = self.guest_to_arbiter.get(idx=0, suffix=0)

        client_cnt = 1  # 1 for guest
        for k, v in shared_dict.items():
            client_cnt += 1
            self.arbiter_to_host.remote(obj=v,
                                        role=consts.HOST, idx=k, suffix=0)
            self.arbiter_to_host.remote(obj=v,
                                        role=consts.HOST, idx=k, suffix=0)

        self.quantizer = QuantizingArbiter(self.int_bits,
                                           self.arbiter_to_guest,
                                           self.guest_to_arbiter,
                                           self.arbiter_to_host,
                                           self.host_to_arbiter,
                                           self.batch,
                                           self.element_bits,
                                           self.secure)
        self.quantizer.broadcast_num_clients(client_cnt)

        return self

    def help_quantize(self):
        if not self.has_set_layer_size_list:
            self.quantizer.set_layer_size_list()
            self.has_set_layer_size_list = True
        return self.quantizer.help_quantize()

    def dynamic_masking(self, masks, total, suffix):
        if not self.mask == "dynamic":
            return
        single_cost = 2 * sum([len(mask) for mask in masks])
        double_cost = 2 * single_cost

        num_clients = len(masks)
        one_hots = []
        for i in range(num_clients):
            one_hot = np.zeros(total, dtype=object)
            one_hot[masks[i]] = 1
            one_hots.append(one_hot)

        canceled_out_pairs = 0
        for i in range(num_clients - 1):
            canceled_out_pairs += sum(one_hots[i] & one_hots[i+1])
        double_cost -= canceled_out_pairs * 2

        # LOGGER.info(f"dynamic masking: single cost {single_cost} double cost {double_cost}")
        # LOGGER.info(f"second {masks[0][:5]}")
        if single_cost <= double_cost:
            d = {"choice": "single", "masks": masks}
        else:
            d = {"choice": "double", "masks": masks}
        LOGGER.info(f"arbiter hint: {d['choice']}")
        self.arbiter_to_guest.remote(obj=d, role=consts.GUEST,
                                     idx=0, suffix=suffix + ("choice",))
        self.arbiter_to_host.remote(obj=d, role=consts.HOST,
                                    idx=-1, suffix=suffix + ("choice",))


class _Client(identify_uuid_sync.Client,
              jzf_dh_keys_exchange_sync.Client):

    def __init__(self, args):
        super(_Client, self).__init__()
        self.cipher = None
        self.quantizer = None

        self.int_bits = args['quantize']['int_bits']

        self.batch = args['quantize']['batch']
        self.element_bits = args['quantize']['element_bits']
        self.padding = args['quantize']['padding']
        self.secure = args['quantize']['secure']
        self.precompute = args['precompute']['enable']
        if self.precompute:
            self.num_params = args['precompute']['num_params']
        if 'mask' in args:
            self.mask = args['mask']
        else:
            self.mask = 'double'

    def encrypt(self, plaintext):
        return self.cipher.encrypt(plaintext)

    def decrypt(self, ciphertext):
        return self.cipher.decrypt(ciphertext)

    def get_idx_list(self):
        return self.cipher.get_idx_list()

    def set_idx_list(self, idx_list):
        self.cipher.set_idx_list(raw_idx_list=idx_list, mode="decrypt")

    def set_iter_index(self, iter_index):
        self.cipher.set_iter_index(iter_index)
        self.quantizer.set_iter(iter_index)

    def unquantize(self, weights):
        return self.quantizer.unquantize(weights)

    def unnormalize(self, weights):
        return self.quantizer.unnormalize(weights)

    def prepare_encrypt(self):
        if not self.precompute:
            return
        else:
            self.cipher.prepare_encrypt()

    def prepare_decrypt(self):
        if not self.precompute:
            return
        else:
            self.cipher.prepare_decrypt()


class Guest(_Client):

    def __init__(self, secure_aggregate_args):
        super(Guest, self).__init__(secure_aggregate_args)
        self.guest_to_arbiter = None
        self.arbiter_to_guest = None
        self.has_send_layer_size_list = False

    def dynamic_masking(self, suffix):
        if not self.mask == "dynamic":
            return
        d = self.arbiter_to_guest.get(idx=0, suffix=suffix + ("choice",))
        LOGGER.info(f"arbiter hint: {d['choice']}")
        self.cipher.masking_scheme = d["choice"]
        self.cipher.masks = d["masks"]

    def create_cipher(self):
        LOGGER.info("synchronizing uuid")
        uuid = self.generate_uuid()
        LOGGER.info(f"local uuid={uuid}")

        LOGGER.info("Diffie-Hellman keys exchanging")
        exchanged_keys = self.key_exchange(uuid)
        LOGGER.info(f"Diffie-Hellman exchanged keys {exchanged_keys}")

        cipher = FlasheCipher(self.int_bits)
        cipher.set_self_uuid(uuid)
        cipher.set_exchanged_keys(exchanged_keys)

        # Guest takes the responsibility to generates the seed
        cipher.generate_prp_seed()
        seed = cipher.get_prp_seed()

        sharing_dict = {}
        aescipher = AESCipher()
        for _, secret_tuple in exchanged_keys.items():
            if secret_tuple[2] == "guest":
                continue

            aescipher.generate_key(256, secret_tuple[1])
            encrypted_seed = aescipher.encrypt(seed)

            sharing_dict[secret_tuple[3]] = encrypted_seed

        del aescipher

        self.guest_to_arbiter.remote(obj=sharing_dict,
                                     role=consts.ARBITER, idx=0, suffix=0)

        self.cipher = cipher
        if self.precompute:
            self.cipher.set_num_params(self.num_params)
            self.cipher.prepare_encrypt()
            # only this need to be done offline
            # other preparation can be done in idle time online

        self.quantizer = QuantizingClient(self.int_bits,
                                          self.arbiter_to_guest,
                                          self.guest_to_arbiter,
                                          self.batch,
                                          self.element_bits,
                                          self.padding,
                                          self.secure)
        num_clients = self.quantizer.receive_num_clients()

        self.cipher.set_num_clients(num_clients)  # for utilizing idle time

        return self

    def register_flashe_cipher(self, transfer_variables):
        self.guest_to_arbiter = transfer_variables.guest_to_arbiter
        self.arbiter_to_guest = transfer_variables.arbiter_to_guest

        self.register_identify_uuid(uuid_transfer_variable=transfer_variables.guest_uuid,
                                    conflict_flag_transfer_variable=transfer_variables.uuid_conflict_flag)
        self.register_dh_key_exchange(dh_pubkey_trv=transfer_variables.dh_pubkey,
                                      dh_ciphertext_trv=transfer_variables.dh_ciphertext_guest,
                                      dh_ciphertext_bc_trv=transfer_variables.dh_ciphertext_bc)
        return self

    def quantize(self, weights):
        if not self.has_send_layer_size_list:
            self.quantizer.send_layer_size_list(weights)
            self.has_send_layer_size_list = True
        return self.quantizer.quantize(weights)

    def normalize(self, weights):
        if not self.has_send_layer_size_list:
            self.quantizer.send_layer_size_list(weights)
            self.has_send_layer_size_list = True
        return self.quantizer.normalize(weights)


class Host(_Client):
    def __init__(self, secure_aggregate_args):
        super(Host, self).__init__(secure_aggregate_args)
        self.arbiter_to_host = None
        self.host_to_arbiter = None
        self.guest_uuid = None
        self.has_set_layer_size_list = False

    def dynamic_masking(self, suffix):
        if not self.mask == "dynamic":
            return
        d = self.arbiter_to_host.get(idx=0, suffix=suffix + ("choice",))
        self.cipher.masking_scheme = d["choice"]
        self.cipher.masks = d["masks"]
        # LOGGER.info(f"third {type(d['masks'])} {d['masks'][0][:5]}")
        LOGGER.info(f"arbiter hint: {d['choice']}")

    def create_cipher(self):
        LOGGER.info("synchronizing uuid")
        uuid = self.generate_uuid()
        LOGGER.info(f"local uuid={uuid}")

        LOGGER.info("Diffie-Hellman keys exchanging")
        exchanged_keys = self.key_exchange(uuid)
        LOGGER.info(f"Diffie-Hellman exchanged keys {exchanged_keys}")

        cipher = FlasheCipher(self.int_bits)
        cipher.set_self_uuid(uuid)
        cipher.set_exchanged_keys(exchanged_keys)

        encrypted_seed = self.arbiter_to_host.get(idx=0, suffix=0)
        secret = exchanged_keys[cipher.get_guest_uuid()][1]
        aescipher = AESCipher()
        aescipher.generate_key(256, secret)
        seed = aescipher.decrypt(encrypted_seed)
        del aescipher

        cipher.generate_prp_seed(assigned_seed=seed)
        self.cipher = cipher
        if self.precompute:
            self.cipher.set_num_params(self.num_params)
            self.cipher.prepare_encrypt()
            # only this need to be done offline
            # other preparation can be done in idle time online

        self.quantizer = QuantizingClient(self.int_bits,
                                          self.arbiter_to_host,
                                          self.host_to_arbiter,
                                          self.batch,
                                          self.element_bits,
                                          self.padding,
                                          self.secure)
        num_clients = self.quantizer.receive_num_clients()

        self.cipher.set_num_clients(num_clients)  # for utilizing idle time

        return self

    def register_flashe_cipher(self, transfer_variables):
        self.host_to_arbiter = transfer_variables.host_to_arbiter
        self.arbiter_to_host = transfer_variables.arbiter_to_host

        self.register_identify_uuid(uuid_transfer_variable=transfer_variables.host_uuid,
                                    conflict_flag_transfer_variable=transfer_variables.uuid_conflict_flag)
        self.register_dh_key_exchange(dh_pubkey_trv=transfer_variables.dh_pubkey,
                                      dh_ciphertext_trv=transfer_variables.dh_ciphertext_host,
                                      dh_ciphertext_bc_trv=transfer_variables.dh_ciphertext_bc)
        return self

    def quantize(self, weights):
        if not self.has_set_layer_size_list:
            self.quantizer.set_layer_size_list(weights)
            self.has_set_layer_size_list = True
        return self.quantizer.quantize(weights)

    def normalize(self, weights):
        if not self.has_set_layer_size_list:
            self.quantizer.set_layer_size_list(weights)
            self.has_set_layer_size_list = True
        return self.quantizer.normalize(weights)
