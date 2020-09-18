# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import collections
import os
import sys
import logging
import struct
import six

import nose
import numpy as np

from test_trt_fp32_helper import TestModelInferenceTrtFp32


class TestHubErnieInferenceTrtFp32(TestModelInferenceTrtFp32):
    """
    TestHubErnieInferenceTrtFp32
    Args:
    Return:
    """

    def __init__(self):
        """__init__
        """
        project_path = os.environ.get("project_path")
        self.model_root = os.path.join(project_path, "Data")
        dy_input_tuple = collections.namedtuple(
            'trt_dynamic_shape_info',
            ['min_input_shape', 'max_input_shape', 'opt_input_shape'])
        min_input_shape = [1, 100, 1]
        max_input_shape = [1, 1024, 1]
        opt_input_shape = [1, 128, 1]
        input_tensor_names = ['input_ids',
                              'segment_ids',
                              'input_mask',
                              'position_ids',
                              'input_ids_2',
                              'segment_ids_2',
                              'input_mask_2',
                              'position_ids_2']
        min_input_shape_dict = {input_tensor_names[x] : min_input_shape for x in range(len(input_tensor_names))}
        max_input_shape_dict = {input_tensor_names[x] : max_input_shape for x in range(len(input_tensor_names))}
        opt_input_shape_dict = {input_tensor_names[x] : opt_input_shape for x in range(len(input_tensor_names))}
        dy_input_infos = dy_input_tuple(min_input_shape_dict, max_input_shape_dict,
                                        opt_input_shape_dict)
        self.trt_dynamic_shape_info = dy_input_infos

    def test_inference_hub_ernie_trt_fp32(self):
        """
        Inference and check value
        hub_ernie trt_fp32 model
        Args:
            None
        Return:
            None
        """
        model_name = "hub-ernie"
        tmp_path = os.path.join(self.model_root, model_name)
        model_path = os.path.join(tmp_path, "hub-ernie-model")
        data_path = os.path.join(tmp_path, "hub-ernie-data", "data.json")
        delta = 0.0001

        res, exp = self.get_infer_results(model_path, data_path)

        for i in range(len(res)):
            self.check_data(res[i].flatten(), exp[i].flatten(), delta)
