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
"""test static name scope."""

import paddle.fluid as fluid


def test_name_scope():
    """
    test name_scope
    :return:
    """
    with fluid.name_scope("s1"):
        a = fluid.data(name='data', shape=[None, 1], dtype='int32')
        b = a + 1
        with fluid.name_scope("s2"):
            c = b * 1
        with fluid.name_scope("s3"):
            d = c / 1
    with fluid.name_scope("s1"):
        f = fluid.layers.pow(d, 2.0)
    with fluid.name_scope("s4"):
        g = f - 1

    # 没有指定的话默认OP在default main program中。
    for op in fluid.default_main_program().block(0).ops:
        # elementwise_add在/s1/中创建
        if op.type == 'elementwise_add':
            assert op.desc.attr("op_namescope") == '/s1/'
        # elementwise_mul在/s1/s2中创建
        elif op.type == 'elementwise_mul':
            assert op.desc.attr("op_namescope") == '/s1/s2/'
        # elementwise_div在/s1/s3中创建
        elif op.type == 'elementwise_div':
            assert op.desc.attr("op_namescope") == '/s1/s3/'
        # elementwise_sum在/s4/中创建
        elif op.type == 'elementwise_sub':
            assert op.desc.attr("op_namescope") == '/s4/'
        # pow在/s1_1/中创建
        elif op.type == 'pow':
            assert op.desc.attr("op_namescope") == '/s1_1/'
