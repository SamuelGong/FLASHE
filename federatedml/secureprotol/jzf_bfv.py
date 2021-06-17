from arch.api.utils import log_utils
from federatedml.secureprotol.encrypt import Encrypt
from Pyfhel import Pyfhel, PyCtxt
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
from memory_tempfile import MemoryTempfile
from functools import reduce
from Crypto.Hash import SHA256
import compress_pickle
import copy

N_JOBS = cpu_count()
LOGGER = log_utils.getLogger()


def _static_add(x, y):
    if isinstance(x, list):
        ret = []
        for xi, yi in zip(x, y):
            ret.append(_static_add(xi, yi))
        return ret
    else:
        return x + y


def _static_encrypt(value, tmp_dir_name, flag_batching=False):
    he = Pyfhel()
    he.restoreContext(tmp_dir_name + "/context")
    he.restorepublicKey(tmp_dir_name + "/pub.key")
    he.restoresecretKey(tmp_dir_name + "/sec.key")
    if flag_batching:
        ciphertext = he.encryptArray(value)
    else:
        ciphertext = he.encryptInt(value)
    return ciphertext.to_bytes()


def _static_decrypt(value, tmp_dir_name, flag_batching=False):
    he = Pyfhel()
    he.restoreContext(tmp_dir_name + "/context")
    he.restorepublicKey(tmp_dir_name + "/pub.key")
    he.restoresecretKey(tmp_dir_name + "/sec.key")
    ciphertext = PyCtxt(pyfhel=he)
    if flag_batching:
        ciphertext.from_bytes(value, 'array')
        # LOGGER.info(f"noise {he.noiseLevel(ciphertext)}")
        return he.decryptBatch(ciphertext)
    else:
        ciphertext.from_bytes(value, 'int')
        return he.decryptInt(ciphertext)


class BFVCipher(Encrypt):
    def __init__(self, p=65537, m=2048, sec=128, flagBatching=False):
        super(BFVCipher, self).__init__()
        self.p = p
        self.m = m
        self.sec = sec
        self.flagBatching = flagBatching
        self.he = Pyfhel()
        self.tempfile = MemoryTempfile()
        self.tmp_dir = self.tempfile.TemporaryDirectory()
        self.key_generated = False
        self.l = None

    def generate_key(self, n=None):
        self.he.contextGen(p=self.p, m=self.m,
                           sec=self.sec, flagBatching=self.flagBatching)
        self.he.keyGen()
        self.he.saveContext(self.tmp_dir.name + "/context")
        self.he.savepublicKey(self.tmp_dir.name + "/pub.key")
        self.he.savesecretKey(self.tmp_dir.name + "/sec.key")
        self.key_generated = True

    def keys_to_bytes(self):
        ret = tuple()
        if self.key_generated is True:
            ret = (
                open(self.tmp_dir.name + "/context", 'rb').read(),
                open(self.tmp_dir.name + "/pub.key", 'rb').read(),
                open(self.tmp_dir.name + "/sec.key", 'rb').read()
            )

            # con_hash = SHA256.new(data=ret[0])
            # pub_hash = SHA256.new(data=ret[1])
            # sec_hash = SHA256.new(data=ret[2])
            # LOGGER.info(f"context hash: {con_hash.hexdigest()}")
            # LOGGER.info(f"pubkey hash: {pub_hash.hexdigest()}")
            # LOGGER.info(f"secret hash: {sec_hash.hexdigest()}")

        return ret

    def bytes_to_keys(self, bytes_tuple):
        open(self.tmp_dir.name + "/context", 'wb').write(bytes_tuple[0])
        open(self.tmp_dir.name + "/pub.key", 'wb').write(bytes_tuple[1])
        open(self.tmp_dir.name + "/sec.key", 'wb').write(bytes_tuple[2])

        # con_hash = SHA256.new(bytes_tuple[0])
        # pub_hash = SHA256.new(bytes_tuple[1])
        # sec_hash = SHA256.new(bytes_tuple[2])
        # LOGGER.info(f"context hash: {con_hash.hexdigest()}")
        # LOGGER.info(f"pubkey hash: {pub_hash.hexdigest()}")
        # LOGGER.info(f"secret hash: {sec_hash.hexdigest()}")

        self.he.restoreContext(self.tmp_dir.name + "/context")
        self.he.restorepublicKey(self.tmp_dir.name + "/pub.key")
        self.he.restoresecretKey(self.tmp_dir.name + "/sec.key")
        self.key_generated = True

    def arbiter_bytes_to_keys(self, bytes):
        open(self.tmp_dir.name + "/context", 'wb').write(bytes)
        self.he.restoreContext(self.tmp_dir.name + "/context")
        self.key_generated = True

    def _multiprocessing_encrypt(self, value, flag_batching=False):
        value_flatten = value.flatten()
        l = len(value_flatten)

        pool_inputs = []
        if flag_batching:
            value_flatten = value_flatten.astype(np.int64)
            m = self.m
            self.l = l  # TODO: currently we only consider flatten_before_encrypt (so there is only one layer)
            batch_num = (l - 1) // m + 1

            for i in range(batch_num):
                begin = i * m
                end = (i + 1) * m
                if end > l:
                    end = l
                pool_inputs.append([value_flatten[begin:end], self.tmp_dir.name, flag_batching])
        else:
            pool_inputs = []
            for i in range(l):
                pool_inputs.append([value_flatten[i], self.tmp_dir.name, flag_batching])

        pool = Pool(N_JOBS)
        ret = pool.starmap(_static_encrypt, pool_inputs)
        pool.close()
        pool.join()
        return ret

    def encrypt(self, value):
        if self.key_generated:
            if self.flagBatching:
                res = self._multiprocessing_encrypt(value, True)
                return res
            else:
                if not isinstance(value, np.ndarray):  # single value
                    return self.he.encryptInt(value)
                else:
                    return self._multiprocessing_encrypt(value, False)
        else:
            return None

    def _multiprocessing_decrypt(self, value, flag_batching=False):
        pool_inputs = []
        for i in range(len(value)):
            pool_inputs.append([value[i], self.tmp_dir.name, flag_batching])

        pool = Pool(N_JOBS)
        ret = pool.starmap(_static_decrypt, pool_inputs)
        pool.close()
        pool.join()

        if flag_batching:
            for idx, e in enumerate(ret):
                ret[idx] = np.array(e, dtype=np.int64)
            ret = reduce(lambda x, y: np.hstack((x, y)), ret)
            ret = ret[:self.l]
            self.l = None
        return ret

    def decrypt(self, value):
        if self.key_generated:
            if self.flagBatching:
                return self._multiprocessing_decrypt(value, True)
            else:
                if not isinstance(value, list):  # single value
                    return self.he.decryptInt(value)
                else:
                    return self._multiprocessing_decrypt(value, False)
        else:
            return None

    def from_bytes(self, arr):
        if isinstance(arr, list):
            ret = []
            for e in arr:
                ret.append(self.from_bytes(e))
            # LOGGER.info(f"{self.he.decryptBatch(ret[0])[:2]}")
            return ret
        else:
            c = PyCtxt(pyfhel=self.he)
            if self.flagBatching:
                c.from_bytes(arr, 'array')
            else:
                c.from_bytes(arr, 'int')
            return c

    def to_bytes(self, arr):
        if isinstance(arr, list):
            ret = []
            for e in arr:
                ret.append(self.to_bytes(e))
            return ret
        else:
            return arr.to_bytes()

    def sum(self, arr):
        arr = self.from_bytes(arr)
        res = reduce(lambda x, y: _static_add(x, y), arr)
        res = self.to_bytes(res)
        return res

    def _noise_level(self, arr):
        if isinstance(arr, list):
            ret = []
            for e in arr:
                ret.append(self._noise_level(e))
            return np.array(ret)
        else:
            c = PyCtxt(pyfhel=self.he)
            if self.flagBatching:
                c.from_bytes(arr, 'array')
            else:
                c.from_bytes(arr, 'int')
            return self.he.noiseLevel(c)

    def noise_level(self, arr):
        noise_levels = self._noise_level(arr).flatten()
        return noise_levels
