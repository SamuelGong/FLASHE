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
import functools
import typing

from arch.api import session
from arch.api.utils.log_utils import LoggerFactory
from arch.api.utils import log_utils
from fate_flow.entity.metric import MetricType, MetricMeta, Metric
from federatedml.framework.homo.procedure import jzf_aggregator as aggregator
from federatedml.model_base import ModelBase
from federatedml.nn.jzf_homo_nn import nn_model
from federatedml.nn.jzf_homo_nn.nn_model import restore_nn_model
from federatedml.optim.convergence import converge_func_factory
from federatedml.param.jzf_homo_nn_param import JZFHomoNNParam as HomoNNParam
from federatedml.transfer_variable.transfer_class.jzf_homo_transfer_variable import JZFHomoTransferVariable as HomoTransferVariable
from federatedml.util import consts

# Logger = LoggerFactory.get_logger()
Logger = log_utils.getLogger()
MODEL_META_NAME = "HomoNNModelMeta"
MODEL_PARAM_NAME = "HomoNNModelParam"

def _build_model_dict(meta, param):
    return {MODEL_META_NAME: meta, MODEL_PARAM_NAME: param}


def _extract_param(model_dict: dict):
    return model_dict.get(MODEL_PARAM_NAME, None)


def _extract_meta(model_dict: dict):
    return model_dict.get(MODEL_META_NAME, None)


class HomoNNBase(ModelBase):

    def __init__(self):
        super().__init__()
        self.model_param = HomoNNParam()
        self.role = None

    def _init_model(self, param):
        super()._init_model(param)
        self.param = param
        self.transfer_variable = HomoTransferVariable()
        self.aggregator = aggregator.with_role(role=self.role,
                                               transfer_variable=self.transfer_variable,
                                               sparsity=param.sparsity,
                                               secure_aggregate=param.secure_aggregate,
                                               secure_aggregate_args=param.secure_aggregate_args)
        self.max_iter = param.max_iter
        self.aggregator_iter = 0

    def _iter_suffix(self):
        return self.aggregator_iter,


class HomoNNArbiter(HomoNNBase):

    def __init__(self):
        super().__init__()
        self.role = consts.ARBITER

    def _init_model(self, param):
        super(HomoNNArbiter, self)._init_model(param)
        early_stop = self.model_param.early_stop
        self.converge_func = converge_func_factory(early_stop.converge_func, early_stop.eps).is_converge
        self.loss_consumed = early_stop.converge_func != "weight_diff"

    def callback_loss(self, iter_num, loss):
        metric_meta = MetricMeta(name='train',
                                 metric_type="LOSS",
                                 extra_metas={
                                     "unit_name": "iters",
                                 })

        self.callback_meta(metric_name='loss', metric_namespace='train', metric_meta=metric_meta)
        self.callback_metric(metric_name='loss',
                             metric_namespace='train',
                             metric_data=[Metric(iter_num, loss)])

    def _check_monitored_status(self):
        loss = self.aggregator.aggregate_loss(suffix=self._iter_suffix())
        Logger.info(f"loss at iter {self.aggregator_iter}: {loss}")
        self.callback_loss(self.aggregator_iter, loss)
        if self.loss_consumed:
            converge_args = (loss,) if self.loss_consumed else (self.aggregator.model,)
            return self.aggregator.send_converge_status(self.converge_func,
                                                        converge_args=converge_args,
                                                        suffix=self._iter_suffix())

    def fit(self, data_inst):
        while self.aggregator_iter < self.max_iter:
            Logger.info(f"begin iter {self.aggregator_iter}")
            self.aggregator.aggregate_and_broadcast(iter_index=self.aggregator_iter,
                                                    suffix=self._iter_suffix())

            if self._check_monitored_status() and False:
                Logger.info(f"early stop at iter {self.aggregator_iter}")
                Logger.info(f"end iter {self.aggregator_iter}")
                break
            Logger.info(f"end iter {self.aggregator_iter}")
            self.aggregator_iter += 1
        else:
            Logger.warn(f"reach max iter: {self.aggregator_iter}, not converged")

    def save_model(self):
        return self.aggregator.model


class HomoNNClient(HomoNNBase):

    def __init__(self):
        super().__init__()

    def _init_model(self, param):
        super(HomoNNClient, self)._init_model(param)
        self.batch_size = param.batch_size
        self.aggregate_every_n_epoch = 1
        self.nn_define = param.nn_define
        self.config_type = param.config_type
        self.optimizer = param.optimizer
        self.loss = param.loss
        self.metrics = param.metrics
        self.data_converter = nn_model.get_data_converter(self.config_type)
        self.model_builder = nn_model.get_nn_builder(config_type=self.config_type)

    def _check_monitored_status(self, data, epoch_degree):
        metrics = self.nn_model.evaluate(data)
        Logger.info(f"metrics at iter {self.aggregator_iter}: {metrics}")
        loss = metrics["loss"]
        self.aggregator.send_loss(loss=loss,
                                  degree=epoch_degree,
                                  suffix=self._iter_suffix())
        return self.aggregator.get_converge_status(suffix=self._iter_suffix())

    def __build_nn_model(self, input_shape):
        self.nn_model = self.model_builder(input_shape=input_shape,
                                           nn_define=self.nn_define,
                                           optimizer=self.optimizer,
                                           loss=self.loss,
                                           metrics=self.metrics)

    def fit(self, data_inst, *args):

        create_label = False
        if self.nn_define["config"]["name"] == "lstm":  # minimal modification as expected
            create_label = True  # due to FATE's limited capability of reading data
        else:
            pass

        data = self.data_converter.convert(data_inst, batch_size=self.batch_size, create_label=create_label)
        # federatedml.nn.backend.tf_keras.nn_model.KerasSequenceData
        # get_shape (x_shape, y_shape), e.g. ((3072,), (10,))
        Logger.info(f"data shape: {data.get_shape()}")
        self.__build_nn_model(data.get_shape()[0])

        epoch_degree = float(len(data))

        while self.aggregator_iter < self.max_iter:
            Logger.info(f"begin iter {self.aggregator_iter}")

            # train
            before = self.nn_model.get_model_weights()
            Logger.info("begin training")
            self.nn_model.train(data, aggregate_every_n_epoch=self.aggregate_every_n_epoch)
            Logger.info("end training")

            # send model for aggregate, then set aggregated model to local
            modify_func: typing.Callable = functools.partial(self.aggregator.aggregate_then_get,
                                                             iter_index=self.aggregator_iter,
                                                             degree=epoch_degree * self.aggregate_every_n_epoch,
                                                             suffix=self._iter_suffix(),
                                                             before=before)
            self.nn_model.modify(modify_func)

            # calc loss and check convergence
            if self._check_monitored_status(data, epoch_degree) and False:
                Logger.info(f"early stop at iter {self.aggregator_iter}")
                Logger.info(f"end iter {self.aggregator_iter}")
                break

            Logger.info(f"end iter {self.aggregator_iter}")
            self.aggregator_iter += 1
        else:
            Logger.warn(f"reach max iter: {self.aggregator_iter}, not converged")

    def export_model(self):
        return _build_model_dict(meta=self._get_meta(), param=self._get_param())

    def _get_meta(self):
        from federatedml.protobuf.generated import jzf_nn_model_meta_pb2
        meta_pb = jzf_nn_model_meta_pb2.JZFNNModelMeta()
        meta_pb.params.CopyFrom(self.model_param.generate_pb())
        meta_pb.aggregate_iter = self.aggregator_iter
        return meta_pb

    def _get_param(self):
        from federatedml.protobuf.generated import jzf_nn_model_param_pb2
        param_pb = jzf_nn_model_param_pb2.JZFNNModelParam()
        param_pb.saved_model_bytes = self.nn_model.export_model()
        return param_pb

    def predict(self, data_inst):
        create_label = False
        if self.nn_define["config"]["name"] == "lstm":  # minimal modification as expected
            create_label = True  # due to FATE's limited capability of reading data
        else:
            pass

        data = self.data_converter.convert(data_inst, batch_size=self.batch_size, create_label=create_label)
        predict = self.nn_model.predict(data)
        num_output_units = predict.shape[1]

        threshold = self.param.predict_param.threshold

        if create_label:
            # num_output_units should not be 1
            # ideal, but trap into dead lock:
            # kv = [(x[0], ([i.argmax() for i in x[1]], [[float(e) for e in i] for i in x[1]])) for x in zip(data.get_keys(), predict)]
            # pred_tbl = session.parallelize(kv, include_key=True)
            # return data_inst.join(pred_tbl,
            #                       lambda d, pred: [d.label, pred[0].item(),
            #                                        [y[x] / sum(y) for x, y in zip(pred[0], pred[1])],
            #                                        {"raw_predict": pred[1]}])

            # side-step (the validation information is non-sense now)
            kv = [(x[0], (x[1][0].argmax(), [float(e) for e in x[1][0]])) for x in
                  zip(data.get_keys(), predict)]
            pred_tbl = session.parallelize(kv, include_key=True)
            return data_inst.join(pred_tbl,
                                  lambda d, pred: [d.label, pred[0].item(),
                                                   pred[1][pred[0]] / sum(pred[1]),
                                                   {"raw_predict": pred[1]}])
        else:
            if num_output_units == 1:
                kv = [(x[0], (0 if x[1][0] <= threshold else 1, x[1][0].item())) for x in zip(data.get_keys(), predict)]
                pred_tbl = session.parallelize(kv, include_key=True)
                return data_inst.join(pred_tbl, lambda d, pred: [d.label, pred[0], pred[1], {"label": pred[0]}])
            else:
                kv = [(x[0], (x[1].argmax(), [float(e) for e in x[1]])) for x in zip(data.get_keys(), predict)]
                pred_tbl = session.parallelize(kv, include_key=True)
                return data_inst.join(pred_tbl,
                                      lambda d, pred: [d.label, pred[0].item(),
                                                       pred[1][pred[0]] / sum(pred[1]),
                                                       {"raw_predict": pred[1]}])

    def load_model(self, model_dict):
        model_dict = list(model_dict["model"].values())[0]
        model_obj = _extract_param(model_dict)
        meta_obj = _extract_meta(model_dict)
        self.model_param.restore_from_pb(meta_obj.params)
        self._init_model(self.model_param)
        self.aggregator_iter = meta_obj.aggregate_iter
        self.nn_model = restore_nn_model(self.config_type, model_obj.saved_model_bytes)


class HomoNNHost(HomoNNClient):

    def __init__(self):
        super().__init__()
        self.role = consts.HOST


class HomoNNGuest(HomoNNClient):

    def __init__(self):
        super().__init__()
        self.role = consts.GUEST
