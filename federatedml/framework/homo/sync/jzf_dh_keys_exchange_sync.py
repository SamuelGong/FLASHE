from federatedml.framework.homo.util.jzf_scatter import Scatter
from federatedml.secureprotol.diffie_hellman import DiffieHellman
from federatedml.transfer_variable.base_transfer_variable import Variable
from federatedml.util import consts


class Arbiter(object):

    def __init__(self):
        super(Arbiter, self).__init__()
        self._dh_pubkey_trv = None
        self._dh_pubkey_scatter = None
        self._dh_ciphertext_bc_trv = None
        self.idx_comm_dict = None
        self.comm_idx_dict = None

    # noinspection PyAttributeOutsideInit
    def register_dh_key_exchange(self,
                                 dh_pubkey_trv: Variable,
                                 dh_ciphertext_host_trv: Variable,
                                 dh_ciphertext_guest_trv: Variable,
                                 dh_ciphertext_bc_trv: Variable):
        self._dh_pubkey_trv = dh_pubkey_trv
        self._dh_pubkey_scatter = Scatter(dh_ciphertext_host_trv, dh_ciphertext_guest_trv)
        self._dh_ciphertext_bc_trv = dh_ciphertext_bc_trv
        return self

    def key_exchange(self):
        p, g = DiffieHellman.key_pair()
        self._dh_pubkey_trv.remote(obj=(int(p), int(g)), role=None, idx=-1)
        tuple_list = self._dh_pubkey_scatter.get()

        pubkey = {}
        idx_comm_dict = {}
        comm_idx_dict = {}
        for cnt, tu in enumerate(tuple_list):
            pubkey[tu[0]] = (cnt,) + tu[1:]
            idx_comm_dict[cnt] = tu[2:]
            comm_idx_dict[tu[2:]] = cnt

        self.idx_comm_dict = idx_comm_dict
        self.comm_idx_dict = comm_idx_dict
        self._dh_ciphertext_bc_trv.remote(obj=pubkey, role=None, idx=-1)


class Client(object):
    def __init__(self):
        super(Client, self).__init__()
        self._dh_pubkey_trv = None
        self._dh_ciphertext_trv = None
        self._dh_ciphertext_bc_trv = None

    # noinspection PyAttributeOutsideInit
    def register_dh_key_exchange(self,
                                 dh_pubkey_trv: Variable,
                                 dh_ciphertext_trv: Variable,
                                 dh_ciphertext_bc_trv: Variable):
        self._dh_pubkey_trv = dh_pubkey_trv
        self._dh_ciphertext_trv = dh_ciphertext_trv
        self._dh_ciphertext_bc_trv = dh_ciphertext_bc_trv
        return self

    def key_exchange(self, uuid):
        p, g = self._dh_pubkey_trv.get(idx=0)
        r = DiffieHellman.generate_secret(p)
        gr = DiffieHellman.encrypt(g, r, p)
        self._dh_ciphertext_trv.remote((uuid, gr),
                                       role=consts.ARBITER, idx=0)

        cipher_texts = self._dh_ciphertext_bc_trv.get(idx=0)
        share_secret = {uid: (tup[0], DiffieHellman.decrypt(
            tup[1], r, p)) + tup[2:] for uid, tup in cipher_texts.items()}
        return share_secret


Guest = Client
Host = Client
