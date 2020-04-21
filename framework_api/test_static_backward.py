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
"""test static backward."""
import numpy as np
import paddle.fluid as fluid
import tools
use_cuda = False


def test_append_backward():
    """
    test_append_backward
    default : loss 、parameter_list = none 、 no_grad_set = NONE 、 callbacks = NONE
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='x', shape=[13], dtype='float32')
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            y_predict = fluid.layers.fc(input=x, size=1, act=None)
            loss = fluid.layers.square_error_cost(input=y_predict, label=y)
            avg_loss = fluid.layers.mean(loss)

            p_g_list1 = fluid.backward.append_backward(loss=avg_loss)
            p_g_list2 = fluid.backward.append_backward(
                loss=avg_loss, parameter_list=[p_g_list1[0][0].name])
            p_g_list3 = fluid.backward.append_backward(
                loss=avg_loss, no_grad_set=set([p_g_list1[0][0].name]))
            p_g_list4 = fluid.backward.append_backward(
                loss=avg_loss,
                parameter_list=[p_g_list1[0][0].name],
                no_grad_set=set([p_g_list1[0][0].name]))
            tools.compare(p_g_list1[0][0].name, "fc_0.w_0")
            tools.compare(p_g_list1[0][1].name, "fc_0.w_0@GRAD")
            tools.compare(p_g_list1[1][0].name, "fc_0.b_0")
            tools.compare(p_g_list1[1][1].name, "fc_0.b_0@GRAD")

            tools.compare(p_g_list2[0][0].name, "fc_0.w_0")
            tools.compare(p_g_list2[0][1].name, "fc_0.w_0@GRAD_0")

            tools.compare(p_g_list3[0][0].name, "fc_0.b_0")
            tools.compare(p_g_list3[0][1].name, "fc_0.b_0@GRAD_1")
            tools.compare(len(p_g_list4), 0)


def test_gradients():
    """
    test_gradients
    default : loss 、parameter_list = none 、 no_grad_set = NONE 、 callbacks = NONE
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.data(name="x", shape=[2, 2], dtype="float32")
            x.stop_gradient = False
            y = fluid.layers.fc(input=x,
                                size=3,
                                param_attr=fluid.initializer.Constant(
                                    value=1.0, force_cpu=True))
            loss = fluid.layers.reduce_sum(y)
            z = fluid.gradients([loss], x)
            tools.compare(z[0].name, "x@GRAD")


def test_gradients_2():
    """
    test_gradients
    default : loss 、parameter_list = none 、 no_grad_set = NONE 、 callbacks = NONE
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.data(name="x", shape=[2, 2], dtype="float32")
            x.stop_gradient = True
            y = fluid.layers.fc(input=x,
                                size=3,
                                param_attr=fluid.initializer.Constant(
                                    value=1.0, force_cpu=True))
            loss = fluid.layers.reduce_sum(y)
            # t = fluid.data(name="t", shape=[1], dtype="float32")
            z = fluid.gradients([loss], x)
            tools.compare(len(z), 0)
