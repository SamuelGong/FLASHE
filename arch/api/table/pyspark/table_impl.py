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

import uuid
from typing import Iterable

from pyspark import SparkContext, RDD

# noinspection PyProtectedMember
from arch.api.table.pyspark import _RDD_ATTR_NAME
from arch.api.table.pyspark import materialize, STORAGE_LEVEL
from arch.api.table.table import Table
from arch.api.utils.profile_util import log_elapsed


class RDDTable(Table):

    # noinspection PyProtectedMember
    @classmethod
    def from_dtable(cls, session_id: str, dtable):
        namespace = dtable._namespace
        name = dtable._name
        partitions = dtable._partitions
        return RDDTable(session_id=session_id, namespace=namespace, name=name, partitions=partitions, dtable=dtable)

    @classmethod
    def from_rdd(cls, rdd: RDD, job_id: str, namespace: str, name: str):
        partitions = rdd.getNumPartitions()
        return RDDTable(session_id=job_id, namespace=namespace, name=name, partitions=partitions, rdd=rdd)

    def __init__(self, session_id: str,
                 namespace: str,
                 name: str = None,
                 partitions: int = 1,
                 rdd: RDD = None,
                 dtable=None):

        self._valid_param_check(rdd, dtable, namespace, partitions)
        setattr(self, _RDD_ATTR_NAME, rdd)
        self._rdd = rdd
        self._partitions = partitions
        self._dtable = dtable
        self.schema = {}
        self._name = name or str(uuid.uuid1())
        self._namespace = namespace
        self._session_id = session_id

    def get_name(self):
        return self._name

    def get_namespace(self):
        return self._namespace

    def __str__(self):
        return f"{self._namespace}, {self._name}, {self._dtable}"

    def __repr__(self):
        return f"{self._namespace}, {self._name}, {self._dtable}"

    def _tmp_table_from_rdd(self, rdd: RDD, name=None):
        """
        tmp table, with namespace == job_id
        """
        rdd = materialize(rdd)
        name = name or str(uuid.uuid1())
        return RDDTable(session_id=self._session_id,
                        namespace=self._namespace,
                        name=name,
                        partitions=rdd.getNumPartitions(),
                        rdd=rdd,
                        dtable=None)

    # self._rdd should not be pickled(spark requires all transformer/action to be invoked in driver).
    def __getstate__(self):
        state = dict(self.__dict__)
        if "_rdd" in state:
            del state["_rdd"]
        return state

    @staticmethod
    def _valid_param_check(rdd, dtable, namespace, partitions):
        assert (rdd is not None) or (dtable is not None), "params rdd and storage are both None"
        assert namespace is not None, "namespace is None"
        assert partitions > 0, "invalid partitions={0}".format(partitions)

    def rdd(self):
        if hasattr(self, "_rdd") and self._rdd is not None:
            return self._rdd

        if self._dtable is None:
            raise AssertionError("try create rdd from None storage")

        return self._rdd_from_dtable()

    # noinspection PyProtectedMember
    @log_elapsed
    def _rdd_from_dtable(self):
        storage_iterator = self._dtable.collect(use_serialize=True)
        if self._dtable.count() <= 0:
            storage_iterator = []

        num_partition = self._dtable._partitions
        self._rdd = SparkContext.getOrCreate() \
            .parallelize(storage_iterator, num_partition) \
            .persist(STORAGE_LEVEL)
        return self._rdd

    def dtable(self):
        """
        rdd -> storage
        """
        if self._dtable:
            return self._dtable
        else:
            if not hasattr(self, "_rdd") or self._rdd is None:
                raise AssertionError("try create dtable from None")
            return self._rdd_to_dtable()

    @log_elapsed
    def _rdd_to_dtable(self, **kwargs):
        self._dtable = self.save_as(name=self._name,
                                    namespace=self._namespace,
                                    partition=self._partitions,
                                    persistent=False)._dtable
        return self._dtable

    def get_partitions(self):
        return self._partitions

    @log_elapsed
    def map(self, func, **kwargs):
        from arch.api.table.pyspark.rdd_func import _map
        rtn_rdd = _map(self.rdd(), func)
        return self._tmp_table_from_rdd(rtn_rdd)

    @log_elapsed
    def mapValues(self, func, **kwargs):
        from arch.api.table.pyspark.rdd_func import _map_value
        rtn_rdd = _map_value(self.rdd(), func)
        return self._tmp_table_from_rdd(rtn_rdd)

    @log_elapsed
    def mapPartitions(self, func, **kwargs):
        from arch.api.table.pyspark.rdd_func import _map_partitions
        rtn_rdd = _map_partitions(self.rdd(), func)
        return self._tmp_table_from_rdd(rtn_rdd)

    @log_elapsed
    def reduce(self, func, **kwargs):
        return self.rdd().values().reduce(func)

    def join(self, other, func=None, **kwargs):
        rdd1 = self.rdd()
        rdd2 = other.rdd()

        @log_elapsed
        def _join(rdda, rddb, **kwargs):
            from arch.api.table.pyspark.rdd_func import _join
            return self._tmp_table_from_rdd(_join(rdda, rddb, func))

        return _join(rdd1, rdd2, **kwargs)

    @log_elapsed
    def glom(self, **kwargs):
        from arch.api.table.pyspark.rdd_func import _glom
        return self._tmp_table_from_rdd(_glom(self.rdd()))

    @log_elapsed
    def sample(self, fraction, seed=None, **kwargs):
        from arch.api.table.pyspark.rdd_func import _sample
        return self._tmp_table_from_rdd(_sample(self.rdd(), fraction, seed))

    @log_elapsed
    def subtractByKey(self, other, **kwargs):
        from arch.api.table.pyspark.rdd_func import _subtract_by_key
        return self._tmp_table_from_rdd(_subtract_by_key(self.rdd(), other.rdd()))

    @log_elapsed
    def filter(self, func, **kwargs):
        from arch.api.table.pyspark.rdd_func import _filter
        return self._tmp_table_from_rdd(_filter(self.rdd(), func))

    @log_elapsed
    def union(self, other, func=lambda v1, v2: v1, **kwargs):
        from arch.api.table.pyspark.rdd_func import _union
        return self._tmp_table_from_rdd(_union(self.rdd(), other.rdd(), func))

    @log_elapsed
    def flatMap(self, func, **kwargs):
        from arch.api.table.pyspark.rdd_func import _flat_map
        return self._tmp_table_from_rdd(_flat_map(self.rdd(), func))

    @log_elapsed
    def collect(self, min_chunk_size=0, use_serialize=True, **kwargs):
        if self._dtable:
            return self._dtable.collect(min_chunk_size, use_serialize)
        else:
            return iter(self.rdd().collect())

    """
    storage api
    """

    def put(self, k, v, use_serialize=True, maybe_large_value=False):
        if not maybe_large_value:
            rtn = self.dtable().put(k, v, use_serialize)
        else:
            from arch.api.table.storage_enhance import split_put
            rtn = split_put(k, v, use_serialize=use_serialize, put_call_back_func=self.dtable().put)
        self._rdd = None
        return rtn

    def put_all(self, kv_list: Iterable, use_serialize=True, chunk_size=100000):
        rtn = self.dtable().put_all(kv_list, use_serialize, chunk_size)
        self._rdd = None
        return rtn

    def get(self, k, use_serialize=True, maybe_large_value=False):
        if not maybe_large_value:
            return self.dtable().get(k, use_serialize)
        else:
            from arch.api.table.storage_enhance import split_get
            return split_get(k=k, use_serialize=use_serialize, get_call_back_func=self.dtable().get)

    def delete(self, k, use_serialize=True):
        rtn = self.dtable().delete(k, use_serialize)
        self._rdd = None
        return rtn

    def destroy(self):
        if self._dtable:
            self._dtable.destroy()
        else:
            self._rdd = None
        return True

    def put_if_absent(self, k, v, use_serialize=True):
        rtn = self.dtable().put_if_absent(k, v, use_serialize)
        self._rdd = None
        return rtn

    # noinspection PyPep8Naming
    def take(self, n=1, keysOnly=False, use_serialize=True):
        if self._dtable:
            return self._dtable.take(n, keysOnly, use_serialize)
        else:
            rtn = self._rdd.take(n)
            if keysOnly:
                rtn = [pair[0] for pair in rtn]
            return rtn

    # noinspection PyPep8Naming
    def first(self, keysOnly=False, use_serialize=True):
        return self.take(1, keysOnly, use_serialize)[0]

    def count(self, **kwargs):
        if self._dtable:
            return self._dtable.count()
        else:
            return self._rdd.count()

    @log_elapsed
    def save_as(self, name, namespace, partition=None, use_serialize=True, persistent=True, **kwargs) -> 'RDDTable':
        partition = partition or self._partitions
        from arch.api import RuntimeInstance
        persistent_engine = RuntimeInstance.SESSION.get_persistent_engine()
        if self._dtable:
            _dtable = self._dtable.save_as(name, namespace, partition,
                                           use_serialize=use_serialize,
                                           persistent_engine=persistent_engine)
            return RDDTable.from_dtable(session_id=self._session_id, dtable=_dtable)
        else:
            from arch.api.table.pyspark.rdd_func import _save_as_func
            return _save_as_func(self._rdd, name=name, namespace=namespace, partition=partition, persistent=persistent)
