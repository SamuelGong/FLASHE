from arch.api.utils import log_utils
from federatedml.framework.homo.sync import identify_uuid_sync, dh_keys_exchange_sync
from federatedml.secureprotol.jzf_paillier import PaillierCipher
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
        self.guest_uuid = None
        self.pubkey_n = None
        self.cipher = None

        self.int_bits = args['quantize']['int_bits']
        self.batch = args['quantize']['batch']
        self.element_bits = args['quantize']['element_bits']
        self.secure = args['quantize']['secure']

        self.quantizer = None
        self.has_set_layer_size_list = False
        self.client_cnt = None

    def set_iter_index(self, iter_index):
        self.quantizer.set_iter(iter_index)

    def register_paillier_cipher(self, transfer_variables):
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

        LOGGER.info("Forwarding Paillier keypairs for Guest to Hosts")
        encrypted_message = self.guest_to_arbiter.get(idx=0, suffix=0)
        self.pubkey_n = encrypted_message[0]
        client_cnt = len(encrypted_message[2].keys()) + 1  # as guest's host_uuid is not in the list
        self.client_cnt = client_cnt

        self.guest_uuid = encrypted_message[1]
        self.arbiter_to_host.remote(obj=encrypted_message[1:], role=consts.HOST,
                                    idx=-1, suffix=0)
        # -1 for all hosts

        self.cipher = PaillierCipher()
        self.cipher.set_n(self.pubkey_n)
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

    def get_n(self):
        return self.cipher.get_n()

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

        self.key_length = args['key_length']
        self.int_bits = args['quantize']['int_bits']
        self.batch = args['quantize']['batch']
        self.element_bits = args['quantize']['element_bits']
        self.padding = args['quantize']['padding']
        self.secure = args['quantize']['secure']

    def encrypt(self, plaintext):
        return self.cipher.encrypt(plaintext)

    def decrypt(self, ciphertext):
        return self.cipher.decrypt(ciphertext)

    def get_n(self):
        return self.cipher.get_n()

    def set_iter_index(self, iter_index):
        self.quantizer.set_iter(iter_index)

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

        cipher = PaillierCipher()
        cipher.set_self_uuid(uuid)
        cipher.set_exchanged_keys(exchanged_keys)

        # Guest takes the responsibility to generates the keys
        cipher.generate_key(self.key_length)
        pub_key, pri_key = cipher.get_key_pair()
        pub_key_raw, pri_key_raw = pickle.dumps(pub_key), pickle.dumps(pri_key)

        sharing_dict = {}
        for host_uuid, secret in exchanged_keys.items():
            if host_uuid == uuid:
                continue

            aescipher = AESCipher()
            aescipher.generate_key(256, secret)
            encrypted_pub_key = aescipher.encrypt(pub_key_raw)
            encrypted_pri_key = aescipher.encrypt(pri_key_raw)

            sharing_dict[host_uuid] = [encrypted_pub_key, encrypted_pri_key]

        del aescipher

        sharing_tuple = (pub_key.n, uuid, sharing_dict)
        self.guest_to_arbiter.remote(obj=sharing_tuple, role=consts.ARBITER, idx=0, suffix=0)

        self.cipher = cipher

        self.quantizer = QuantizingClient(self.int_bits,
                                          self.arbiter_to_guest,
                                          self.guest_to_arbiter,
                                          self.batch,
                                          self.element_bits,
                                          self.padding,
                                          self.secure)
        self.quantizer.receive_num_clients()
        return self

    def register_paillier_cipher(self, transfer_variables):
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

        cipher = PaillierCipher()
        cipher.set_self_uuid(uuid)
        cipher.set_exchanged_keys(exchanged_keys)

        message = self.arbiter_to_host.get(idx=0, suffix=0)
        guest_uuid = message[0]
        self.guest_uuid = guest_uuid
        secret = exchanged_keys[guest_uuid]

        encrypted_list = message[1][uuid]
        aescipher = AESCipher()
        aescipher.generate_key(256, secret)
        pub_key, pri_key = aescipher.decrypt(encrypted_list[0]), aescipher.decrypt(encrypted_list[1])
        pub_key, pri_key = pickle.loads(pub_key), pickle.loads(pri_key)

        del aescipher
        cipher.set_public_key(pub_key)
        cipher.set_privacy_key(pri_key)

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

    def register_paillier_cipher(self, transfer_variables):
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