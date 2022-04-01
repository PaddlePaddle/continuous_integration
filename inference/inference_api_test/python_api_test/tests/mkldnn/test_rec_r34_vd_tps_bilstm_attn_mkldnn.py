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

import pytest
import nose
import numpy as np

from test_mkldnn_helper import TestModelInferenceMKLDNN

TestBase = TestModelInferenceMKLDNN(data_path="Data")

@pytest.mark.p1
def test_inference_rec_r34_vd_tps_bilstm_attn_mkldnn():
    """
    Inference and check value
    rec_r34_vd_tps_bilstm_attn mkldnn model
    Args:
        None
    Returns:
        None
    """
    model_name = "rec_r34_vd_tps_bilstm_attn"
    tmp_path = os.path.join(TestBase.model_root, "python-ocr-infer")
    model_path = os.path.join(tmp_path, model_name)
    data_path = os.path.join(tmp_path, "word_rec_data_3_32_100", "data.json")
    delta = 0.0001

    res, exp = TestBase.get_infer_results(model_path, data_path)

    for i in range(len(res)):
        TestBase.check_data(res[i].flatten(), exp[i].flatten(), delta)


