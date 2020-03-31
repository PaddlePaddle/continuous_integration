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
"""significant bugfix testcase."""
import paddle.fluid as fluid
import numpy
import tools


def test_scalar():
    """
    test scalar bug fix, input lr(float) but it is be set as double in the backend, fix this bug
    Returns:
        None
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            train_data=numpy.array([[1.0], [2.0], [3.0], [4.0]]).astype('float32')
            y_true = numpy.array([[2.0], [4.0], [6.0], [8.0]]).astype('float32')
            lr = fluid.layers.data(name="lr", shape=[1], dtype='float32', append_batch_size=False)
            x = fluid.data(name="x", shape=[None, 1], dtype='float32')
            y = fluid.data(name="y", shape=[None, 1], dtype='float32')
            y_predict = fluid.layers.fc(input=x, size=1, act=None)
            cost = fluid.layers.square_error_cost(input=y_predict,label=y)
            avg_cost = fluid.layers.mean(cost)
            sgd_optimizer = fluid.optimizer.Adam(learning_rate=lr)
            sgd_optimizer.minimize(avg_cost)
            cpu = fluid.CPUPlace()
            exe = fluid.Executor(cpu)
            exe.run(startup_program)
            res = exe.run(
                    feed={'x':train_data, 'y':y_true, 'lr': numpy.asarray([1], dtype=numpy.float32)},
                    fetch_list=[y_predict, avg_cost])

            expect = [104.31773]
            tools.compare(res[1], expect)

