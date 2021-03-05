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

import pytest
import nose
import numpy as np

from test_mkldnn_helper import TestModelInferenceMKLDNN

TestBase = TestModelInferenceMKLDNN(data_path="Data")

@pytest.mark.p0
def test_inference_hub_ernie_mkldnn():
    """
    Inference and check value
    hub_ernie mkldnn model
    Args:
        None
    Return:
        None
    """
    model_name = "hub-ernie"
    tmp_path = os.path.join(TestBase.model_root, model_name)
    model_path = os.path.join(tmp_path, "hub-ernie-model")
    data_path = os.path.join(tmp_path, "hub-ernie-data", "data.json")
    delta = 0.0001

    res, exp = TestBase.get_infer_results(model_path, data_path)

    for i in range(len(res)):
        TestBase.check_data(res[i].flatten(), exp[i].flatten(), delta)

