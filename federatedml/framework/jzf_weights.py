#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import abc
import operator
import numpy as np
from arch.api.utils import log_utils
from arch.api.utils.splitable import segment_transfer_enabled
from federatedml.secureprotol.encrypt import Encrypt
from multiprocessing import cpu_count, Pool
N_JOBS = cpu_count()
# N_JOBS = 1  # empirically efficient
LOGGER = log_utils.getLogger()


def chunks_idx(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield si, si+(d+1 if i < r else d)


def _to_bytes_old(flatten_array, num_bits):
    res = 0
    l = len(flatten_array)
    for element in flatten_array:
        res <<= num_bits
        res += element
    return res, l


def _to_bytes(flatten_array, num_bits):
    lcm = np.lcm(num_bits, 8)
    batch_size = lcm // num_bits
    num_bytes = lcm // 8

    l = len(flatten_array)
    if isinstance(flatten_array, np.ndarray):
        flatten_array = flatten_array.astype(object)
    batch_num = (l - 1) // batch_size + 1
    s = None
    for i in range(batch_num - 1):
        begin = i * batch_size
        end = (i + 1) * batch_size

        ss = 0
        for j in range(begin, end):
            ss <<= num_bits
            ss += flatten_array[j]
        #             print('\t', j, ss, ss.to_bytes(num_bytes, 'big'))

        if s is None:
            s = ss.to_bytes(num_bytes, 'big')
        else:
            s += ss.to_bytes(num_bytes, 'big')
    #         print(s)

    i = batch_num - 1
    begin = i * batch_size
    end = min((i + 1) * batch_size, l)
    ss = 0
    for j in range(begin, end):
        ss <<= num_bits
        ss += flatten_array[j]

    s = int.from_bytes(s, 'big')
    #     print(s)
    s <<= ((end - begin) * num_bits)
    s += ss
    # LOGGER.info(f"{s.bit_length()}")
    return s, l


def _from_bytes_old(big_int, _len, num_bits):
    ret = []

    mask = (1 << num_bits) - 1
    for i in range(_len):
        ret.append(big_int & mask)
        big_int >>= num_bits

    return ret


def _from_bytes(s, l, num_bits):
    lcm = np.lcm(num_bits, 8)
    batch_size = lcm // num_bits
    num_bytes = lcm // 8

    batch_num = (l - 1) // batch_size + 1

    i = batch_num - 1
    begin = i * batch_size
    end = min((i + 1) * batch_size, l)

    result = []
    # the .item() here is extremely IMPORTANT!
    # otherwise the bit width of left shift will be constrained to 32/64
    tail = s & ((1 << ((end - begin) * num_bits).item()) - 1)

    s >>= ((end - begin) * num_bits)

    mask = (1 << num_bits) - 1
    for j in range(begin, end):
        result.append(tail & mask)
        a = tail & mask
        tail >>= num_bits
        b = tail & mask
        # LOGGER.info(f"{j} {a} {b} {tail}")


    # LOGGER.info(f"{num_bytes * (batch_num - 1)}, {s.bit_length()}")
    s = s.to_bytes(num_bytes * (batch_num - 1), 'big')
    for i in reversed(range(batch_num - 1)):
        begin = i * num_bytes
        end = (i + 1) * num_bytes

        ss = s[begin:end]
        ss = int.from_bytes(ss, 'big')
        for j in range(batch_size):
            result.append(ss & mask)
            ss >>= num_bits

    return result


class JZFTransferableWeights(metaclass=segment_transfer_enabled()):
    def __init__(self, weights, bits, need_compress, shape, cls, *args, **kwargs):
        self._bits = bits
        self._weights = weights
        self._cls = cls
        self._shape = shape

        if args:
            self._args = args
        if kwargs:
            self._kwargs = kwargs

        if self._bits is not None and need_compress:
            self.compress()

    def compress(self):  # use multiprocessing to accelerate
        bits = self._bits
        weights = self._weights
        res = {}

        # first save the shape information
        # then convert
        self._shape = {}
        for k in weights.keys():
            self._shape[k] = weights[k].shape
            flatten_layer = weights[k].flatten()
            l = len(flatten_layer)

            pool_inputs = []
            sizes = []
            pool = Pool(N_JOBS)
            for begin, end in chunks_idx(range(l), N_JOBS):
                sizes.append(end - begin)

                pool_inputs.append([flatten_layer[begin:end], bits])

            pool_outputs = pool.starmap(_to_bytes, pool_inputs)
            pool.close()
            pool.join()

            s = 0
            for idx, output in enumerate(pool_outputs):
                s += output[0] << (int(np.sum(sizes[idx + 1:])) * bits)
            # num_bytes = (bits * l - 1) // 8 + 1
            # res[k] = s.to_bytes(num_bytes, 'big')

            # t = _to_bytes(flatten_layer, bits)
            # s = t[0]
            # LOGGER.info(f"{s.bit_length()}")
            # LOGGER.info(f"{s & (1 << (5449480 - 1)) == 0}")
            # LOGGER.info(f"{s & (1 << (5449480 - 2)) == 0}")
            # LOGGER.info(f"{s & (1 << (5449480 - 3)) == 0}")
            # LOGGER.info(f"{s & (1 << (5449480 - 4)) == 0}")
            res[k] = s

        self._weights = res

    def decompress(self):
        bits = self._bits
        weights = self._weights
        shape = self._shape

        new_weights = {}
        for k in weights.keys():
            # _bytes = weights[k]
            _shape = shape[k]
            _len = int(np.prod(_shape))
            # big_int = int.from_bytes(_bytes, 'big')
            big_int = weights[k]

            # pool_inputs = []
            # pool = Pool(N_JOBS)
            # for begin, end in chunks_idx(range(_len), N_JOBS):
            #     mask = (1 << ((end - begin) * bits)) - 1
            #     int_fragment = (big_int >> (begin * bits)) & mask
            #     pool_inputs.append([int_fragment, end - begin, bits])
            #
            # pool_outputs = pool.starmap(_from_bytes, pool_inputs)
            # pool.close()
            # pool.join()
            #
            # res = []
            # for output in pool_outputs:
            #     res += output

            res = _from_bytes(big_int, _len, bits)

            res.reverse()
            res = np.array(res).reshape(_shape).astype(object)
            new_weights[k] = res

        return new_weights

    def with_degree(self, degree):
        setattr(self, "_degree", degree)
        return self

    def get_degree(self, default=None):
        return getattr(self, "_degree", default)

    def with_idx_list(self, idx_list):
        setattr(self, "_idx_list", idx_list)
        return self

    def get_idx_list(self, default=None):
        return getattr(self, "_idx_list", default)

    def with_shape_list(self, shape_list):  # this is for partition for pure Paillier
        setattr(self, "_shape_list", shape_list)
        return self

    def get_shape_list(self, default=None):
        return getattr(self, "_shape_list", default)

    def get_unboxed(self, need_decompress=False):
        if self._bits is None:
            return self._weights
        else:
            if need_decompress:
                return self.decompress()
            else:
                return self._weights

    def get_weights(self, need_decompress=False):

        size_dict = None
        if self._bits is None:
            w = self._weights
            need_compress_for_next_remote = False
        else:
            if need_decompress:
                need_compress_for_next_remote = True
                w = self.decompress()
            else:
                need_compress_for_next_remote = False
                w = self._weights
                size_dict = {}
                for k in self._shape.keys():
                    size_dict[k] = int(np.prod(self._shape[k]))

        if not hasattr(self, "_args") and not hasattr(self, "_kwargs"):
            return self._cls(w, bits=self._bits,
                             need_compress_for_next_remote=need_compress_for_next_remote,
                             size_dict=size_dict, shape=self._shape)
        else:
            args = self._args if hasattr(self, "_args") else ()
            kwargs = self._kwargs if hasattr(self, "_kwargs") else {}
            return self._cls(w, bits=self._bits,
                             need_compress_for_next_remote=need_compress_for_next_remote,
                             size_dict=size_dict, shape=self._shape,
                             *args, **kwargs)

    unboxed = property(get_unboxed)
    weights = property(get_weights)


class JZFWeights(object):

    def __init__(self, l, bits=None, need_compress_for_next_remote=False,
                 size_dict=None, shape=None):
        self._weights = l
        self._bits = bits
        self._need_compress_for_next_remote = need_compress_for_next_remote
        self._size_dict = size_dict
        self._shape = shape

    def set_bits(self, bits):
        self._bits = bits
        self._need_compress_for_next_remote = True

    def get_bits(self):
        return self._bits

    def for_remote(self):
        return JZFTransferableWeights(self._weights, self._bits,
                                      self._need_compress_for_next_remote, self._shape,
                                      self.__class__)

    @property
    def unboxed(self):
        return self._weights

    @abc.abstractmethod
    def map_values(self, func, inplace):
        pass

    @abc.abstractmethod
    def binary_op(self, other, func, inplace):
        pass

    @abc.abstractmethod
    def axpy(self, a, y):
        pass

    def decrypted(self, cipher: Encrypt, inplace=True):
        return self.map_values(cipher.decrypt, inplace=inplace)

    def encrypted(self, cipher: Encrypt, inplace=True):
        return self.map_values(cipher.encrypt, inplace=inplace)

    def __mod__(self, other):
        return self.map_values(lambda x: x % other, inplace=False)

    def __imul__(self, other):
        return self.map_values(lambda x: x * other, inplace=True)

    # def __mul__(self, other):
    #     return self.map_values(lambda x: x * other, inplace=False)

    def __mul__(self, other):
        return self.binary_op(other, operator.mul, inplace=False)

    def __iadd__(self, other):
        return self.binary_op(other, operator.add, inplace=True)

    def __add__(self, other):
        # LOGGER.debug("In binary_op0, _w: {}".format(self._weights))
        return self.binary_op(other, operator.add, inplace=False)

    def __isub__(self, other):
        return self.binary_op(other, operator.sub, inplace=True)

    def __sub__(self, other):
        return self.binary_op(other, operator.sub, inplace=False)

    def __truediv__(self, other):
        return self.map_values(lambda x: x / other, inplace=False)

    def __itruediv__(self, other):
        return self.map_values(lambda x: x / other, inplace=True)


class JZFNumericWeights(JZFWeights):
    def __init__(self, v):
        super().__init__(v)

    def map_values(self, func, inplace):
        v = func(self._weights)
        if inplace:
            self._weights = v
            return self
        else:
            return JZFNumericWeights(v)

    def binary_op(self, other: 'JZFNumpyWeights', func, inplace):
        v = func(self._weights, other._weights)
        if inplace:
            self._weights = v
            return self
        else:
            return JZFNumericWeights(v)

    def axpy(self, a, y: 'JZFNumpyWeights'):
        self._weights = self._weights + a * y._weights
        return self


# for loss
class JZFNumpyWeights(JZFWeights):
    def __init__(self, arr):
        super().__init__(arr)

    def map_values(self, func, inplace):
        if inplace:
            size = self._weights.size
            view = self._weights.view().reshape(size)
            for i in range(size):
                view[i] = func(view[i])
            return self
        else:
            vec_func = np.vectorize(func)
            weights = vec_func(self._weights)
            return JZFNumpyWeights(weights)

    def binary_op(self, other: 'JZFNumpyWeights', func, inplace):
        if inplace:
            size = self._weights.size
            view = self._weights.view().reshape(size)
            view_other = other._weights.view().reshpae(size)
            for i in range(size):
                view[i] = func(view[i], view_other[i])
            return self
        else:
            vec_func = np.vectorize(func)
            weights = vec_func(self._weights, other._weights)
            return JZFNumpyWeights(weights)

    def axpy(self, a, y: 'JZFNumpyWeights'):
        size = self._weights.size
        view = self._weights.view().reshape(size)
        view_other = y._weights.view().reshpae(size)
        for i in range(size):
            view[i] += a * view_other[i]
        return self


class JZFOrderDictWeights(JZFWeights):

    def __init__(self, d, bits=None, need_compress_for_next_remote=False,
                 size_dict=None, shape=None):
        super().__init__(d, bits, need_compress_for_next_remote, size_dict, shape)
        self.walking_order = sorted(d.keys(), key=str)

    def refresh_walking_order(self):
        self.walking_order = sorted(self._weights.keys(), key=str)

    def map_values(self, func, inplace):
        if inplace:
            for k in self.walking_order:
                self._weights[k] = func(self._weights[k])
            return self
        else:
            _w = dict()
            for k in self.walking_order:
                _w[k] = func(self._weights[k])
            return JZFOrderDictWeights(_w, bits=self._bits,
                                       need_compress_for_next_remote=self._need_compress_for_next_remote,
                                       size_dict=self._size_dict,
                                       shape=self._shape)

    def binary_op(self, other: 'JZFOrderDictWeights', func, inplace):
        if inplace:
            for k in self.walking_order:
                self._weights[k] = func(other._weights[k], self._weights[k])
            return self
        else:
            _w = dict()
            for k in self.walking_order:
                _w[k] = func(other._weights[k], self._weights[k])
            return JZFOrderDictWeights(_w, bits=self._bits,
                                       need_compress_for_next_remote=self._need_compress_for_next_remote,
                                       size_dict=self._size_dict,
                                       shape=self._shape)

    def axpy(self, a, y: 'JZFOrderDictWeights'):
        for k in self.walking_order:
            self._weights[k] += a * y._weights[k]
        return self
