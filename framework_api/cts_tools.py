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
"""cts test tools"""

import numpy as np
from nose import tools


def check_data(result, expect, delta=None):
    """
    比对结果
    :param result: 输出结果  numpy
    :param expect: 预期结果  array
    :param delta: 精度误差值
    :return:
    """
    if delta:
        for i in range(len(expect)):
            tools.assert_almost_equal(
                result[i], np.float32(expect[i]), delta=delta)
    else:
        for i in range(len(expect)):
            tools.assert_equal(result[i], np.float32(expect[i]))
