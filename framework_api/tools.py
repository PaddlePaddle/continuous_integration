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

from nose import tools
import numpy as np
import math


def compare(input, expect, delta=1e-8):
    """
    比较函数
    :param input: 输入值
    :param expect: 输出值
    :param delta: 误差值
    :return:
    """
    if type(input) == np.ndarray:
        expect = np.array(expect)
        res = np.allclose(input, expect, atol=delta)
        # 出错打印错误数据
        if res is False:
            print("the input is {}".format(input))
            print("the expect is {}".format(expect))
        tools.assert_true(res)
        tools.assert_equal(input.shape, expect.shape)
    elif type(input) == list:
        input = np.array(input)
        expect = np.array(expect)
        res = np.allclose(input, expect, atol=delta)
        # 出错打印错误数据
        if res is False:
            print("the input is {}".format(input))
            print("the expect is {}".format(expect))
        tools.assert_true(res)
        tools.assert_equal(input.shape, expect.shape)
    elif type(input) == str:
        res = input == expect
        if res is False:
            print("the input is {}".format(input))
            print("the expect is {}".format(expect))
        tools.assert_true(res)
    else:
        tools.assert_almost_equal(input, expect, delta=delta)


def sigmoid(x):
    """
    sigmoid
    :param x: (float)
    :return:
    """
    return 1 / (1 + math.exp(-x))


