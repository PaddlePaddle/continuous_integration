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
import os
import sys
import argparse
import logging
import struct
import six

import nose
import numpy as np

from test_trt_fp32_helper import TestModelInferenceTrtFp32


class TestMaskRcnnInferenceTrtFp32(TestModelInferenceTrtFp32):
    """
    TestModelInferenceTrtFp32
    Args:
    Return:
    """

    def test_inference_mask_rcnn_trt_fp32(self):
        """
        Inference and check value
        mask_rcnn trt_fp32 model
        Args:
            None
        Return:
            None
        """
        model_name = "mask_rcnn_r50_1x"
        tmp_path = os.path.join(self.model_root, "Detection")
        model_path = os.path.join(tmp_path, model_name, "model")
        data_path = os.path.join(tmp_path, model_name, "data/data.json")
        delta = 0.001
        min_subgraph_size = 40

        res, exp = self.get_infer_results(model_path, data_path,
                                          min_subgraph_size)

        for i in range(len(res)):
            self.check_data(res[i].flatten(), exp[i].flatten(), delta)
