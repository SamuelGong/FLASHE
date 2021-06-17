import numpy as np

class TwoComplement(object):

    def __init__(self, int_bits):
        super(TwoComplement, self).__init__()

    @classmethod
    def true_to_two(cls, value, int_bits):
        mod = 2 ** int_bits
        value = value % mod
        return value

    @classmethod
    def two_to_true(cls, value, int_bits):
        border = 2 ** (int_bits - 1)
        offset = - 2 ** int_bits
        ret = np.where(value < border, value, value + offset)
        return ret
