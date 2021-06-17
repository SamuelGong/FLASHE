from arch.api.utils import log_utils
from federatedml.secureprotol.encrypt import Encrypt
import tenseal as ts
import numpy as np
from functools import reduce

LOGGER = log_utils.getLogger()


class CKKSCipher(Encrypt):
    def __init__(self, poly_modulus_degree=8192,
                 coeff_mod_bit_sizes=None,
                 global_scale=2**40):
        super(CKKSCipher, self).__init__()

        self.poly_modulus_degree = poly_modulus_degree
        if coeff_mod_bit_sizes:
            self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        else:
            self.coeff_mod_bit_sizes = []  # should be this since we do no do any multiplication
        self.global_scale = global_scale

        self.context = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
            encryption_type=ts.ENCRYPTION_TYPE.SYMMETRIC
        )
        self.context.generate_galois_keys()
        self.context.global_scale = self.global_scale

    def from_bytes(self, arr):
        if isinstance(arr, list):
            ret = []
            for e in arr:
                ret.append(self.from_bytes(e))
            return ret
        else:
            c = ts.CKKSVector.load(self.context, arr)
            return c

    def to_bytes(self, arr):
        if isinstance(arr, list):
            ret = []
            for e in arr:
                ret.append(self.to_bytes(e))
            return ret
        else:
            return arr.serialize()

    def encrypt(self, value):
        # value should be a 1-d np.array
        return ts.ckks_vector(self.context, value).serialize()

    def sum(self, arr):
        loaded = [ts.CKKSVector.load(self.context, e) for e in arr]
        res = reduce(lambda x, y: x + y, loaded)
        return res.serialize()

    def decrypt(self, value):
        return np.array(ts.CKKSVector.load(self.context, value).decrypt())

    def encrypt_no_batch(self, value):
        return [ts.ckks_vector(self.context, [i]).serialize() for i in value]

    def sum_no_batch(self, arr):
        l = len(arr[0])
        result = []
        for i in range(l):
            scalars = [ts.CKKSVector.load(self.context, e[i]) for e in arr]
            result.append(reduce(lambda x, y: x + y, scalars).serialize())
        return result

    def decrypt_no_batch(self, value):
        return np.array([ts.CKKSVector.load(self.context, i).decrypt() for i in value])

    def set_context(self, bytes):
        context = ts.Context.load(bytes)
        self.context = context

    def get_context(self, save_secret_key=False):
        return self.context.serialize(save_secret_key=save_secret_key,
                                      save_galois_keys=False,
                                      save_relin_keys=False)
