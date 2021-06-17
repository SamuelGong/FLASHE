from arch.api.utils import log_utils
from federatedml.framework.homo.sync import identify_uuid_sync, dh_keys_exchange_sync
from federatedml.secureprotol.jzf_bfv import BFVCipher
from federatedml.secureprotol.jzf_aes import AESCipher
from federatedml.secureprotol.jzf_quantize import QuantizingClient, QuantizingArbiter
from federatedml.util import consts
import pickle

LOGGER = log_utils.getLogger()


class Arbiter(identify_uuid_sync.Arbiter,
              dh_keys_exchange_sync.Arbiter):

    def __init__(self, args):
        super(Arbiter, self).__init__()
        self.arbiter_to_guest = None
        self.arbiter_to_host = None
        self.guest_to_arbiter = None
        self.host_to_arbiter = None
        self.client_cnt = None

        self.p = args['p']
        self.m = args['m']
        self.sec = args['sec']
        self.flagBatching = args['flagBatching']

        self.int_bits = args['quantize']['int_bits']
        self.batch = args['quantize']['batch']
        self.element_bits = args['quantize']['element_bits']
        self.secure = args['quantize']['secure']

        self.quantizer = None
        self.has_set_layer_size_list = False
        self.client_cnt = None

        self.cipher = None

    def set_iter_index(self, iter_index):
        self.quantizer.set_iter(iter_index)

    def register_bfv_cipher(self, transfer_variables):
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

        cipher = BFVCipher(self.p, self.m,
                           self.sec, self.flagBatching)

        LOGGER.info("Forwarding BFV context for Guest to Hosts")
        encrypted_message = self.guest_to_arbiter.get(idx=0, suffix=0)
        self.arbiter_to_host.remote(obj=encrypted_message, role=consts.HOST,
                                    idx=-1, suffix=0)  # -1 for all hosts

        context_raw = encrypted_message[1]
        cipher.arbiter_bytes_to_keys(context_raw)
        self.cipher = cipher
        client_cnt = len(encrypted_message[2].keys()) + 1  # as guest's host_uuid is not in the list
        self.client_cnt = client_cnt

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


class _Client(identify_uuid_sync.Client,
              dh_keys_exchange_sync.Client):

    def __init__(self, args):
        super(_Client, self).__init__()
        self.cipher = None
        self.quantizer = None

        # on encryption
        self.p = args['p']
        self.m = args['m']
        self.sec = args['sec']
        self.flagBatching = args['flagBatching']

        # on batching
        self.int_bits = args['quantize']['int_bits']
        self.batch = args['quantize']['batch']
        self.element_bits = args['quantize']['element_bits']
        self.padding = args['quantize']['padding']
        self.secure = args['quantize']['secure']

    def set_iter_index(self, iter_index):
        self.quantizer.set_iter(iter_index)

    def encrypt(self, plaintext):
        return self.cipher.encrypt(plaintext)

    def decrypt(self, ciphertext):
        return self.cipher.decrypt(ciphertext)

    def unquantize(self, weights):
        return self.quantizer.unquantize(weights)

    def unnormalize(self, weights):
        return self.quantizer.unnormalize(weights)


class Guest(_Client):
    def __init__(self, secure_aggregate_args):
        super(Guest, self).__init__(secure_aggregate_args)
        self.guest_to_arbiter = None
        self.arbiter_to_guest = None
        self.has_send_layer_size_list = False

    def create_cipher(self):
        LOGGER.info("synchronizing uuid")
        uuid = self.generate_uuid()
        LOGGER.info(f"local uuid={uuid}")

        LOGGER.info("Diffie-Hellman keys exchanging")
        exchanged_keys = self.key_exchange(uuid)
        LOGGER.info(f"Diffie-Hellman exchanged keys {exchanged_keys}")

        cipher = BFVCipher(self.p, self.m,
                           self.sec, self.flagBatching)

        # Guest takes the responsibility to generates the keys
        cipher.generate_key()
        self.cipher = cipher
        context_raw, pub_key_raw, pri_key_raw = cipher.keys_to_bytes()

        sharing_dict = {}
        for host_uuid, secret in exchanged_keys.items():
            if host_uuid == uuid:
                continue
            aescipher = AESCipher()
            aescipher.generate_key(256, secret)
            encrypted_pub_key = aescipher.encrypt(pub_key_raw)
            encrypted_pri_key = aescipher.encrypt(pri_key_raw)
            sharing_dict[host_uuid] = [
                encrypted_pub_key,
                encrypted_pri_key
            ]
        del aescipher

        sharing_tuple = (uuid, context_raw, sharing_dict)
        self.guest_to_arbiter.remote(obj=sharing_tuple, role=consts.ARBITER, idx=0, suffix=0)

        self.quantizer = QuantizingClient(self.int_bits,
                                          self.arbiter_to_guest,
                                          self.guest_to_arbiter,
                                          self.batch,
                                          self.element_bits,
                                          self.padding,
                                          self.secure)

        self.quantizer.receive_num_clients()
        return self

    def register_bfv_cipher(self, transfer_variables):
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

    def create_cipher(self):
        LOGGER.info("synchronizing uuid")
        uuid = self.generate_uuid()
        LOGGER.info(f"local uuid={uuid}")

        LOGGER.info("Diffie-Hellman keys exchanging")
        exchanged_keys = self.key_exchange(uuid)
        LOGGER.info(f"Diffie-Hellman exchanged keys {exchanged_keys}")

        cipher = BFVCipher(self.p, self.m,
                           self.sec, self.flagBatching)

        message = self.arbiter_to_host.get(idx=0, suffix=0)
        guest_uuid = message[0]
        secret = exchanged_keys[guest_uuid]
        context = message[1]
        encrypted_list = message[2][uuid]

        aescipher = AESCipher()
        aescipher.generate_key(256, secret)
        pub_key = aescipher.decrypt(encrypted_list[0])
        pri_key = aescipher.decrypt(encrypted_list[1])
        del aescipher
        cipher.bytes_to_keys((context, pub_key, pri_key))
        self.cipher = cipher

        self.quantizer = QuantizingClient(self.int_bits,
                                          self.arbiter_to_host,
                                          self.host_to_arbiter,
                                          self.batch,
                                          self.element_bits,
                                          self.padding,
                                          self.secure)
        self.quantizer.receive_num_clients()
        return self

    def register_bfv_cipher(self, transfer_variables):
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