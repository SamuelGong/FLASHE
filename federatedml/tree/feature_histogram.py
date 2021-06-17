#!/usr/bin/env python    
# -*- coding: utf-8 -*- 

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
################################################################################
#
#
################################################################################

# =============================================================================
# FeatureHistogram
# =============================================================================
import functools
import copy
import numpy as np
from arch.api.utils import log_utils
from federatedml.feature.fate_element_type import NoneType

LOGGER = log_utils.getLogger()


class FeatureHistogram(object):
    def __init__(self):
        pass

    @staticmethod
    def accumulate_histogram(histograms):
        for i in range(len(histograms)):
            for j in range(len(histograms[i])):
                for k in range(1, len(histograms[i][j])):
                    for r in range(len(histograms[i][j][k])):
                        histograms[i][j][k][r] += histograms[i][j][k - 1][r]

        return histograms

    @staticmethod
    def calculate_histogram(data_bin, grad_and_hess,
                            bin_split_points, bin_sparse_points,
                            valid_features=None, node_map=None,
                            use_missing=False, zero_as_missing=False):
        LOGGER.info("bin_shape is {}, node num is {}".format(bin_split_points.shape, len(node_map)))
        batch_histogram_cal = functools.partial(
            FeatureHistogram.batch_calculate_histogram,
            bin_split_points=bin_split_points, bin_sparse_points=bin_sparse_points,
            valid_features=valid_features, node_map=node_map,
            use_missing=use_missing, zero_as_missing=zero_as_missing)

        agg_histogram = functools.partial(FeatureHistogram.aggregate_histogram, node_map=node_map)

        batch_histogram = data_bin.join(grad_and_hess, \
                                        lambda data_inst, g_h: (data_inst, g_h)).mapPartitions(batch_histogram_cal)

        return batch_histogram.reduce(agg_histogram)

    @staticmethod
    def aggregate_histogram(batch_histogram1, batch_histogram2, node_map=None):
        for i in range(len(batch_histogram1)):
            for j in range(len(batch_histogram1[i])):
                for k in range(len(batch_histogram1[i][j])):
                    for r in range(len(batch_histogram1[i][j][k])):
                        batch_histogram1[i][j][k][r] += batch_histogram2[i][j][k][r]

        return batch_histogram1

    @staticmethod
    def batch_calculate_histogram(kv_iterator, bin_split_points=None,
                                  bin_sparse_points=None, valid_features=None,
                                  node_map=None, use_missing=False, zero_as_missing=False):
        data_bins = []
        node_ids = []
        grad = []
        hess = []

        data_record = 0

        for _, value in kv_iterator:
            data_bin, nodeid_state = value[0]
            unleaf_state, nodeid = nodeid_state
            if unleaf_state == 0 or nodeid not in node_map:
                continue
            g, h = value[1]
            data_bins.append(data_bin)
            node_ids.append(nodeid)
            grad.append(g)
            hess.append(h)

            data_record += 1

        LOGGER.info("begin batch calculate histogram, data count is {}".format(data_record))
        node_num = len(node_map)

        missing_bin = 1 if use_missing else 0
        zero_optim = [[[0 for i in range(3)]
                       for j in range(bin_split_points.shape[0])]
                      for k in range(node_num)]
        zero_opt_node_sum = [[0 for i in range(3)]
                             for j in range(node_num)]

        node_histograms = []
        for k in range(node_num):
            feature_histogram_template = []
            for fid in range(bin_split_points.shape[0]):
                if valid_features is not None and valid_features[fid] is False:
                    feature_histogram_template.append([])
                    continue
                else:
                    feature_histogram_template.append([[0 for i in range(3)]
                                                       for j in range(bin_split_points[fid].shape[0] + 1 + missing_bin)])

            node_histograms.append(feature_histogram_template)

        assert len(feature_histogram_template) == bin_split_points.shape[0]

        for rid in range(data_record):
            nid = node_map.get(node_ids[rid])
            zero_opt_node_sum[nid][0] += grad[rid]
            zero_opt_node_sum[nid][1] += hess[rid]
            zero_opt_node_sum[nid][2] += 1
            for fid, value in data_bins[rid].features.get_all_data():
                if valid_features is not None and valid_features[fid] is False:
                    continue

                if use_missing and value == NoneType():
                    value = -1

                node_histograms[nid][fid][value][0] += grad[rid]
                node_histograms[nid][fid][value][1] += hess[rid]
                node_histograms[nid][fid][value][2] += 1

                zero_optim[nid][fid][0] += grad[rid]
                zero_optim[nid][fid][1] += hess[rid]
                zero_optim[nid][fid][2] += 1

        for nid in range(node_num):
            for fid in range(bin_split_points.shape[0]):
                if valid_features is not None and valid_features[fid] is True:
                    if not use_missing or (use_missing and not zero_as_missing):
                        sparse_point = bin_sparse_points[fid]
                        node_histograms[nid][fid][sparse_point][0] += zero_opt_node_sum[nid][0] - zero_optim[nid][fid][0]
                        node_histograms[nid][fid][sparse_point][1] += zero_opt_node_sum[nid][1] - zero_optim[nid][fid][1]
                        node_histograms[nid][fid][sparse_point][2] += zero_opt_node_sum[nid][2] - zero_optim[nid][fid][2]
                    else:
                        node_histograms[nid][fid][-1][0] += zero_opt_node_sum[nid][0] - zero_optim[nid][fid][0]
                        node_histograms[nid][fid][-1][1] += zero_opt_node_sum[nid][1] - zero_optim[nid][fid][1]
                        node_histograms[nid][fid][-1][2] += zero_opt_node_sum[nid][2] - zero_optim[nid][fid][2]

        return node_histograms
