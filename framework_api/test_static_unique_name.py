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
"""test static unique name."""

import paddle.fluid as fluid
import tools


def test_generate():
    """
    test generate
    :return:
    """
    with fluid.unique_name.guard():
        name1 = fluid.unique_name.generate('fc')
        name2 = fluid.unique_name.generate('fc')
        name3 = fluid.unique_name.generate("fc")
        tools.compare(name1, "fc_0")
        tools.compare(name2, "fc_1")
        tools.compare(name3, "fc_2")


def test_guard():
    """
    test guard
    :return:
    """
    with fluid.unique_name.guard():
        name1 = fluid.unique_name.generate('fc')
    with fluid.unique_name.guard():
        name2 = fluid.unique_name.generate('fc')
        name3 = fluid.unique_name.generate('fc')
    tools.compare(name1, "fc_0")
    tools.compare(name2, "fc_0")
    tools.compare(name3, "fc_1")


def test_guard1():
    """
    test guard with new_generator= A and B
    :return:
    """
    with fluid.unique_name.guard(new_generator='A'):
        name1 = fluid.unique_name.generate('fc')
        name2 = fluid.unique_name.generate('fc')
    with fluid.unique_name.guard(new_generator='B'):
        name3 = fluid.unique_name.generate('fc')
        name4 = fluid.unique_name.generate('fc')
    tools.compare(name1, "Afc_0")
    tools.compare(name2, "Afc_1")
    tools.compare(name3, "Bfc_0")
    tools.compare(name4, "Bfc_1")


def test_switch():
    """
    test switch
    :return:
    """
    name1 = fluid.unique_name.generate('fc')
    name2 = fluid.unique_name.generate('fc')
    # print(name1)
    tools.compare(name1, "fc_0")
    tools.compare(name2, "fc_1")
    pre_generator = fluid.unique_name.switch()
    name2 = fluid.unique_name.generate('fc')
    tools.compare(name2, "fc_0")
    fluid.unique_name.switch(pre_generator[0])
    name3 = fluid.unique_name.generate('fc')
    tools.compare(name3, "fc_2")


def test_switch1():
    """
    test switch with new_generator = pre
    :return:
    """
    with fluid.unique_name.guard(new_generator='A'):
        name1 = fluid.unique_name.generate('fc')
        name2 = fluid.unique_name.generate('fc')
        tools.compare(name1, "Afc_0")
        tools.compare(name2, "Afc_1")
        pre = fluid.unique_name.switch()
        name3 = fluid.unique_name.generate("fc")
        tools.compare(name3, "fc_0")
        fluid.unique_name.switch(pre[0])
        name4 = fluid.unique_name.generate("fc")
        tools.compare(name4, "Afc_2")
