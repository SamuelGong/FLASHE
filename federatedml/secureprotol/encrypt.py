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

import numpy as np
from collections import Iterable


class Encrypt(object):
    def __init__(self):
        self.public_key = None
        self.privacy_key = None

    def generate_key(self, n_length=0):
        pass

    def set_public_key(self, public_key):
        pass

    def get_public_key(self):
        pass

    def set_privacy_key(self, privacy_key):
        pass

    def get_privacy_key(self):
        pass

    def encrypt(self, value):
        pass

    def decrypt(self, value):
        pass

    def encrypt_list(self, values):
        result = [self.encrypt(msg) for msg in values]
        return result

    def decrypt_list(self, values):
        result = [self.decrypt(msg) for msg in values]
        return result

    def distribute_decrypt(self, X):
        decrypt_table = X.mapValues(lambda x: self.decrypt(x))
        return decrypt_table

    def distribute_encrypt(self, X):
        encrypt_table = X.mapValues(lambda x: self.encrypt(x))
        return encrypt_table

    def _recursive_func(self, obj, func):
        if isinstance(obj, np.ndarray):
            if len(obj.shape) == 1:
                return np.reshape([func(val) for val in obj], obj.shape)
            else:
                return np.reshape([self._recursive_func(o, func) for o in obj], obj.shape)
        elif isinstance(obj, Iterable):
            return type(obj)(
                self._recursive_func(o, func) if isinstance(o, Iterable) else func(o) for o in obj)
        else:
            return func(obj)

    def recursive_encrypt(self, X):
        return self._recursive_func(X, self.encrypt)

    def recursive_decrypt(self, X):
        return self._recursive_func(X, self.decrypt)
