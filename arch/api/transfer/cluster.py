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

import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, Tuple

# noinspection PyPackageRequirements
import grpc

from arch.api.proto import federation_pb2, federation_pb2_grpc
from arch.api.proto.federation_pb2 import TransferMeta, TransferDataDesc
from arch.api.transfer import Party, Federation
from arch.api.transfer import Rubbish
from arch.api.utils import eggroll_serdes
from arch.api.utils.log_utils import getLogger
from arch.api.utils.splitable import maybe_split_object, is_split_head, split_get
# noinspection PyProtectedMember
from eggroll.api.cluster.eggroll import _DTable, _EggRoll
from eggroll.api.proto import basic_meta_pb2, storage_basic_pb2

_ser_des = eggroll_serdes.PickleSerdes

OBJECT_STORAGE_NAME = "__federation__"

ERROR_STATES = [federation_pb2.CANCELLED, federation_pb2.ERROR]
REMOTE_FRAGMENT_OBJECT_USE_D_TABLE = False

LOGGER = getLogger()


def _await_ready(receive_func, check_func, transfer_meta):
    resp_meta = receive_func(transfer_meta)
    while resp_meta.transferStatus != federation_pb2.COMPLETE:
        if resp_meta.transferStatus in ERROR_STATES:
            raise IOError(
                "receive terminated, state: {}".format(federation_pb2.TransferStatus.Name(resp_meta.transferStatus)))
        resp_meta = check_func(resp_meta)
    return resp_meta


def _thread_receive(receive_func, check_func, name, tag, session_id, src_party, dst_party):

    my_log_msg = f"src={src_party}, dst={dst_party}, name={name}, tag={tag}"
    LOGGER.info(f"[GET] begin {my_log_msg}")

    job = basic_meta_pb2.Job(jobId=session_id, name=name)
    transfer_meta = TransferMeta(job=job, tag=tag, src=src_party.to_pb(), dst=dst_party.to_pb(),
                                 type=federation_pb2.RECV)
    recv_meta = _await_ready(receive_func, check_func, transfer_meta)
    desc = recv_meta.dataDesc

    if desc.transferDataType == federation_pb2.DTABLE:
        LOGGER.debug(
            f"[GET] table ready: src={src_party}, dst={dst_party}, name={name}, tag={tag}, session_id={session_id}")
        table = _get_table(desc)
        return table, table, None

    if desc.transferDataType == federation_pb2.OBJECT:
        obj_table = _cache_get_obj_storage_table[src_party]
        __tagged_key = _ser_des.deserialize(desc.taggedVariableName)

        obj = obj_table.get(__tagged_key)

        if not is_split_head(obj):
            return obj, (obj_table, __tagged_key), None

        num_split = obj.num_split()
        LOGGER.debug(f"[GET] num_fragments={num_split}: {my_log_msg}")
        fragment_keys = []
        fragment_table = obj_table
        if REMOTE_FRAGMENT_OBJECT_USE_D_TABLE:
            LOGGER.debug(f"[GET] getting fragments table: {my_log_msg}")
            job = basic_meta_pb2.Job(jobId=session_id, name=name)
            transfer_meta = TransferMeta(job=job, tag=f"{tag}.fragments_table", src=src_party.to_pb(),
                                         dst=dst_party.to_pb(), type=federation_pb2.RECV)
            _resp_meta = _await_ready(receive_func, check_func, transfer_meta)
            table = _get_table(_resp_meta.dataDesc)
            fragment_table = table
            fragment_keys.extend(list(range(num_split)))
        else:
            for i in range(num_split):
                LOGGER.debug(f"[GET] getting fragments({i + 1}/{num_split}): {my_log_msg}")
                job = basic_meta_pb2.Job(jobId=session_id, name=name)
                transfer_meta = TransferMeta(job=job, tag=_fragment_tag(tag, i), src=src_party.to_pb(),
                                             dst=dst_party.to_pb(), type=federation_pb2.RECV)
                _resp_meta = _await_ready(receive_func, check_func, transfer_meta)
                LOGGER.debug(f"[GET] fragments({i + 1}/{num_split}) ready: {my_log_msg}")
                __fragment_tagged_key = _ser_des.deserialize(_resp_meta.dataDesc.taggedVariableName)
                fragment_keys.append(__fragment_tagged_key)
        LOGGER.debug(f"[GET] large object ready: {my_log_msg}")
        return obj, (obj_table, __tagged_key), (fragment_table, fragment_keys)
    else:
        raise IOError(f"unknown transfer type: {recv_meta.dataDesc.transferDataType}")


def _fragment_tag(tag, idx):
    return f"{tag}.__frag__{idx}"


def _get_table(transfer_data_desc):
    name = transfer_data_desc.storageLocator.name
    namespace = transfer_data_desc.storageLocator.namespace
    persistent = transfer_data_desc.storageLocator.type != storage_basic_pb2.IN_MEMORY
    return _EggRoll.get_instance().table(name=name, namespace=namespace, persistent=persistent)


# noinspection PyProtectedMember
def _get_storage_locator(table):
    return storage_basic_pb2.StorageLocator(type=table._type,
                                            namespace=table._namespace,
                                            name=table._name,
                                            fragment=table._partitions)


def _create_table(name, namespace, persistent=True):
    return _EggRoll.get_instance().table(name=name, namespace=namespace, persistent=persistent)


def _create_fragment_obj_table(namespace, persistent=True):
    eggroll = _EggRoll.get_instance()
    name = eggroll.generateUniqueId()
    return eggroll.table(name=name, namespace=namespace, persistent=persistent)


_cache_remote_obj_storage_table = {}
_cache_get_obj_storage_table = {}


def _get_obj_storage_table_name(src, dst):
    return f"{OBJECT_STORAGE_NAME}.{src.role}-{src.party_id}-{dst.role}-{dst.party_id}"


def _fill_cache(parties, local, session_id):
    for party in parties:
        if party == local:
            continue
        _cache_get_obj_storage_table[party] = _create_table(_get_obj_storage_table_name(party, local), session_id)
        _cache_remote_obj_storage_table[party] = _create_table(_get_obj_storage_table_name(local, party), session_id)

class FederationRuntime(Federation):

    def __init__(self, session_id, runtime_conf, host, port):
        super().__init__(session_id, runtime_conf)

        channel = grpc.insecure_channel(target=f"{host}:{port}",
                                        options=[('grpc.max_send_message_length', -1),
                                                 ('grpc.max_receive_message_length', -1)])
        self._stub = federation_pb2_grpc.TransferSubmitServiceStub(channel)
        self.__pool = ThreadPoolExecutor()

        # init object storage tables
        _fill_cache(self.all_parties, self.local_party, self._session_id)

    def remote(self, obj, name: str, tag: str, parties: Union[Party, list]) -> Rubbish:
        if isinstance(parties, Party):
            parties = [parties]
        self._remote_side_auth(name=name, parties=parties)
        rubbish = Rubbish(name, tag)

        # if obj is a dtable, remote it
        if isinstance(obj, _DTable):
            obj.set_gc_disable()
            for party in parties:
                my_log_msg = f"src={self.local_party}, dst={party}, name={name}, tag={tag}"
                LOGGER.info(f"[REMOTE] dtable begin {my_log_msg}")
                self._send(transfer_type=federation_pb2.DTABLE, name=name, tag=tag, dst_party=party, rubbish=rubbish,
                           table=obj)
                LOGGER.info(f"[REMOTE] dtable end {my_log_msg}")

            return rubbish

        # if obj is object, put it in specified dtable, then remote the dtable
        first, fragments = maybe_split_object(obj)
        num_fragment = len(fragments)

        if REMOTE_FRAGMENT_OBJECT_USE_D_TABLE and num_fragment > 1:
            fragment_storage_table = _create_fragment_obj_table(self._session_id)
            fragment_storage_table.put_all(fragments)

        for party in parties:
            my_log_msg = f"src={self.local_party}, dst={party}, name={name}, tag={tag}"
            LOGGER.info(f"[REMOTE] object begin {my_log_msg}")
            obj_table = _cache_remote_obj_storage_table[party]
            LOGGER.debug(f"[REMOTE] get obj_table")
            # remote object or remote fragment header
            self._send(transfer_type=federation_pb2.OBJECT, name=name, tag=tag, dst_party=party, rubbish=rubbish,
                       table=obj_table, obj=first)

            # remote fragments
            # impl 1
            if REMOTE_FRAGMENT_OBJECT_USE_D_TABLE and num_fragment > 1:
                LOGGER.debug(f"[REMOTE] sending fragment table: {my_log_msg}")
                # noinspection PyUnboundLocalVariable
                self._send(transfer_type=federation_pb2.DTABLE, name=name, tag=f"{tag}.fragments_table",
                           dst_party=party, rubbish=rubbish, table=fragment_storage_table)
                LOGGER.debug(f"[REMOTE] done fragment table: {my_log_msg}")
            # impl 2
            elif not REMOTE_FRAGMENT_OBJECT_USE_D_TABLE:
                for fragment_index, fragment in fragments:
                    LOGGER.debug(f"[REMOTE] sending fragment({fragment_index}/{num_fragment}): {my_log_msg}")
                    self._send(transfer_type=federation_pb2.OBJECT, name=name, tag=_fragment_tag(tag, fragment_index),
                               dst_party=party, rubbish=rubbish, table=obj_table, obj=fragment)
                    LOGGER.debug(f"[REMOTE] done fragment({fragment_index}/{num_fragment}): {my_log_msg}")

            LOGGER.info(f"[REMOTE] object end {my_log_msg}")

        return rubbish

    def _check_get_status_async(self, name: str, tag: str, parties: list) -> dict:
        self._get_side_auth(name=name, parties=parties)

        futures = {
            self.__pool.submit(_thread_receive, self._stub.recv, self._stub.checkStatus, name, tag,
                               self._session_id, party,
                               self.local_party): party
            for party in parties}
        return futures

    def async_get(self, name: str, tag: str, parties: list) -> typing.Generator:
        rubbish = Rubbish(name, tag)
        futures = self._check_get_status_async(name, tag, parties)
        for future in as_completed(futures):
            party = futures[future]
            LOGGER.debug(f"[ASYNC] {party} begin")
            obj, head, frags = future.result()
            LOGGER.debug(f"[ASYNC] {party} get result")
            if isinstance(obj, _DTable):
                rubbish.add_table(obj)
                LOGGER.debug(f"[ASYNC] {party} is dtable")
                yield (party, obj)
            else:
                table, key = head
                LOGGER.debug(f"[ASYNC] {party} else")
                rubbish.add_obj(table, key)
                LOGGER.debug(f"[ASYNC] {party} finish add object")
                if not is_split_head(obj):
                    LOGGER.debug(f"[ASYNC] {party} not split head")
                    yield (party, obj)
                else:
                    frag_table, frag_keys = frags
                    LOGGER.debug(f"[ASYNC] {party} else else")
                    rubbish.add_table(frag_table)
                    LOGGER.debug(f"[ASYNC] {party} finish add_table")
                    fragments = [frag_table.get(key) for key in frag_keys]
                    LOGGER.debug(f"[ASYNC] {party} get fragments")
                    yield (party, split_get(fragments))
        LOGGER.debug("[ASYNC] yield rubbish")
        yield (None, rubbish)

    def get(self, name: str, tag: str, parties: Union[Party, list]) -> Tuple[list, Rubbish]:
        if isinstance(parties, Party):
            parties = [parties]
        rtn = {}
        rubbish = None

        for p, v in self.async_get(name, tag, parties):
            if p is not None:
                my_log_msg = f"src={p}, dst={self.local_party}, name={name}, tag={tag}"
                rtn[p] = v
                LOGGER.info(f"[GET] end {my_log_msg}")
            else:
                rubbish = v

        LOGGER.debug("[GET] return")
        return [rtn[p] for p in parties], rubbish

    def _send(self, transfer_type, name: str, tag: str, dst_party: Party, rubbish: Rubbish, table: _DTable, obj=None):
        tagged_key = f"{name}-{tag}"
        if transfer_type == federation_pb2.OBJECT:
            LOGGER.debug("[REMOTE] pb2")
            table.put(tagged_key, obj)
            LOGGER.debug("[REMOTE] table put finished")
            rubbish.add_obj(table, tagged_key)
            LOGGER.debug("[REMOTE] rubbish add_obj finished")
        else:
            LOGGER.debug("[REMOTE] no pb2")
            rubbish.add_table(table)
            LOGGER.debug("[REMOTE] rubbish add_table finished")
        data_desc = TransferDataDesc(transferDataType=transfer_type,
                                     storageLocator=_get_storage_locator(table),
                                     taggedVariableName=_ser_des.serialize(tagged_key))
        LOGGER.debug("[REMOTE] get data_desc")
        job = basic_meta_pb2.Job(jobId=self._session_id, name=name)
        LOGGER.debug("[REMOTE] get job")
        transfer_meta = TransferMeta(job=job, tag=tag, src=self.local_party.to_pb(), dst=dst_party.to_pb(),
                                     dataDesc=data_desc, type=federation_pb2.SEND)
        LOGGER.debug("[REMOTE] get transfer_meta")
        self._stub.send(transfer_meta)
        LOGGER.debug("[REMOTE] _stub send finished")
