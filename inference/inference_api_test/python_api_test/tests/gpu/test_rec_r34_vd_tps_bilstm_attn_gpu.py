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

from test_gpu_helper import TestModelInferenceGPU


class TestRecR34VdTpsBiLstmAttnInferenceGPU(TestModelInferenceGPU):
    """
    TestModelInferenceGPU
    Args:
    Return:
    """

    def __init__(self):
        """__init__
        """
        project_path = os.environ.get("project_path")
        self.model_root = os.path.join(project_path, "Data")

    def test_inference_rec_r34_vd_tps_bilstm_attn_gpu(self):
        """
        Inference and check value
        rec_r34_vd_tps_bilstm_attn gpu model
        Args:
            None
        Return:
            None
        """
        model_name = "rec_r34_vd_tps_bilstm_attn"
        tmp_path = os.path.join(self.model_root, "python-ocr-infer")
        model_path = os.path.join(tmp_path, model_name)
        data_path = os.path.join(tmp_path, "word_rec_data_3_32_100", "data.json")
        delta = 0.0009

        res, exp = self.get_infer_results(model_path, data_path)

        for i in range(len(res)):
            self.check_data(res[i].flatten(), exp[i].flatten(), delta)
