from arch.api.utils import log_utils
from federatedml.framework.homo.sync import identify_uuid_sync, dh_keys_exchange_sync
from federatedml.secureprotol.jzf_ckks import CKKSCipher
from federatedml.secureprotol.jzf_aes import AESCipher
from federatedml.util import consts
from federatedml.framework.weights import DictWeights

LOGGER = log_utils.getLogger()


class Arbiter(identify_uuid_sync.Arbiter,
              dh_keys_exchange_sync.Arbiter):

    def __init__(self, args):
        super(Arbiter, self).__init__()
        self.arbiter_to_guest = None
        self.arbiter_to_host = None
        self.guest_to_arbiter = None
        self.host_to_arbiter = None

        self.poly_modulus_degree = args['poly_modulus_degree']
        self.cipher = None

    def set_iter_index(self, iter_index):
        pass

    def register_ckks_cipher(self, transfer_variables):
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

        cipher = CKKSCipher(poly_modulus_degree=self.poly_modulus_degree)

        LOGGER.info("Forwarding CKKS context for Guest to Hosts")
        message = self.guest_to_arbiter.get(idx=0, suffix=0)
        self.arbiter_to_host.remote(obj=message, role=consts.HOST,
                                    idx=-1, suffix=0)  # -1 for all hosts

        context_raw = message.weights._weights['context_raw_for_arbiter']
        cipher.set_context(context_raw)
        self.cipher = cipher

        return self

    def help_quantize(self):
        pass


class _Client(identify_uuid_sync.Client,
              dh_keys_exchange_sync.Client):

    def __init__(self, args):
        super(_Client, self).__init__()
        self.poly_modulus_degree = args['poly_modulus_degree']
        self.cipher = None

    def set_iter_index(self, iter_index):
        pass

    def encrypt(self, plaintext):
        return self.cipher.encrypt(plaintext)

    def decrypt(self, ciphertext):
        return self.cipher.decrypt(ciphertext)

    def unquantize(self, weights):
        return weights

    def unnormalize(self, weights):
        return weights

    def quantize(self, weights):
        return weights

    def normalize(self, weights):
        return weights


class Guest(_Client):
    def __init__(self, secure_aggregate_args):
        super(Guest, self).__init__(secure_aggregate_args)
        self.guest_to_arbiter = None
        self.arbiter_to_guest = None

    def create_cipher(self):
        LOGGER.info("synchronizing uuid")
        uuid = self.generate_uuid()
        LOGGER.info(f"local uuid={uuid}")

        LOGGER.info("Diffie-Hellman keys exchanging")
        exchanged_keys = self.key_exchange(uuid)
        LOGGER.info(f"Diffie-Hellman exchanged keys {exchanged_keys}")

        cipher = CKKSCipher(poly_modulus_degree=self.poly_modulus_degree)
        self.cipher = cipher

        # Guest takes the responsibility to generates the keys
        context_raw_for_arbiter = self.cipher.get_context()
        context_for_hosts = self.cipher.get_context(True)

        host_dict = {}
        for host_uuid, secret in exchanged_keys.items():
            if host_uuid == uuid:
                continue
            aescipher = AESCipher()
            aescipher.generate_key(256, secret)
            context_for_hosts_encrypted = aescipher.encrypt(context_for_hosts)
            host_dict[host_uuid] = context_for_hosts_encrypted
        del aescipher

        sharing_dict = {}
        sharing_dict['uuid'] = uuid
        sharing_dict['context_raw_for_arbiter'] = context_raw_for_arbiter
        sharing_dict['sharing_dict'] = host_dict
        sharing_weight = DictWeights(sharing_dict).for_remote()

        self.guest_to_arbiter.remote(obj=sharing_weight,
                                     role=consts.ARBITER,
                                     idx=0, suffix=0)
        return self

    def register_ckks_cipher(self, transfer_variables):
        self.guest_to_arbiter = transfer_variables.guest_to_arbiter
        self.arbiter_to_guest = transfer_variables.arbiter_to_guest

        self.register_identify_uuid(uuid_transfer_variable=transfer_variables.guest_uuid,
                                    conflict_flag_transfer_variable=transfer_variables.uuid_conflict_flag)
        self.register_dh_key_exchange(dh_pubkey_trv=transfer_variables.dh_pubkey,
                                      dh_ciphertext_trv=transfer_variables.dh_ciphertext_guest,
                                      dh_ciphertext_bc_trv=transfer_variables.dh_ciphertext_bc)
        return self


class Host(_Client):

    def __init__(self, secure_aggregate_args):
        super(Host, self).__init__(secure_aggregate_args)
        self.arbiter_to_host = None
        self.host_to_arbiter = None

    def create_cipher(self):
        LOGGER.info("synchronizing uuid")
        uuid = self.generate_uuid()
        LOGGER.info(f"local uuid={uuid}")

        LOGGER.info("Diffie-Hellman keys exchanging")
        exchanged_keys = self.key_exchange(uuid)
        LOGGER.info(f"Diffie-Hellman exchanged keys {exchanged_keys}")

        cipher = CKKSCipher(poly_modulus_degree=self.poly_modulus_degree)

        message = self.arbiter_to_host.get(idx=0, suffix=0)
        guest_uuid = message.weights._weights['uuid']
        secret = exchanged_keys[guest_uuid]
        context_for_hosts_encrypted = message.weights._weights['sharing_dict'][uuid]

        aescipher = AESCipher()
        aescipher.generate_key(256, secret)
        context_for_hosts = aescipher.decrypt(context_for_hosts_encrypted)
        del aescipher

        cipher.set_context(context_for_hosts)
        self.cipher = cipher
        return self

    def register_ckks_cipher(self, transfer_variables):
        self.host_to_arbiter = transfer_variables.host_to_arbiter
        self.arbiter_to_host = transfer_variables.arbiter_to_host

        self.register_identify_uuid(uuid_transfer_variable=transfer_variables.host_uuid,
                                    conflict_flag_transfer_variable=transfer_variables.uuid_conflict_flag)
        self.register_dh_key_exchange(dh_pubkey_trv=transfer_variables.dh_pubkey,
                                      dh_ciphertext_trv=transfer_variables.dh_ciphertext_host,
                                      dh_ciphertext_bc_trv=transfer_variables.dh_ciphertext_bc)
        return self