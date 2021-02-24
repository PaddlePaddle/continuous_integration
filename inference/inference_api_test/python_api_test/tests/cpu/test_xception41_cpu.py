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

from test_cpu_helper import TestModelInferenceCPU

TestBase = TestModelInferenceCPU()

@pytest.mark.p0
def test_inference_xception41_cpu():
    """
    Inference and check value
    xception41 cpu model
    Args:
        None
    Return:
        None
    """
    model_name = "Xception_41_pretrained"
    tmp_path = os.path.join(TestBase.model_root, "classification")
    model_path = os.path.join(tmp_path, model_name, "model")
    data_path = os.path.join(tmp_path, model_name, "data/data.json")
    delta = 0.0001

    res, exp = TestBase.get_infer_results(model_path, data_path)

    for i in range(len(res)):
        TestBase.check_data(res[i].flatten(), exp[i].flatten(), delta)
