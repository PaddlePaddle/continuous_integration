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
"""test static optimizer."""
import numpy as np
import paddle as paddle
import paddle.fluid as fluid
import tools
use_cuda = False


#  SGDOptimizer
def test_SGDOptimizer():
    """
    test SGDOptimizer
    default : lr = 0.001 、 regularization = NONE 、 name = NONE
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.992, 0.992, 0.992],
                        [0.98800004, 0.98800004, 0.98800004]]
            expect_b = [-0.004, -0.004, -0.004]
            expect_res = [29.831999]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_SGDOptimizer_lr_float():
    """
    test SGDOptimizer : learning_rate_float=0.000001
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.000001)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.999992, 0.999992, 0.999992],
                        [0.99998796, 0.99998796, 0.99998796]]
            expect_b = [-4.e-06, -4.e-06, -4.e-06]
            expect_res = [29.999832]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_SGDOptimizer_lr_var():
    """
    test SGDOptimizer : learning_rate_variable=0.000001
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            lr = fluid.layers.create_global_var(
                shape=[1],
                value=0.000001,
                dtype='float32',
                persistable=True,
                name="lr")
            optimizer = fluid.optimizer.SGDOptimizer(learning_rate=lr)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.999992, 0.999992, 0.999992],
                        [0.99998796, 0.99998796, 0.99998796]]
            expect_b = [-4.e-06, -4.e-06, -4.e-06]
            expect_res = [29.999832]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_SGDOptimizer_regularization():
    """
    test SGDOptimizer : regularization:fluid.regularizer.L2Decay(0.01)
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.SGDOptimizer(
                learning_rate=0.001,
                regularization=fluid.regularizer.L2Decay(
                    regularization_coeff=0.01))
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(
                    train_program,
                    feed={"inp": np_inp},
                    fetch_list=["fc.w_0@GRAD", "fc.b_0@GRAD", out.name])
            expect_w = [[4.0099597, 4.0099597, 4.0099597],
                        [6.0099397, 6.0099397, 6.0099397]]
            expect_b = [1.99998, 1.99998, 1.99998]
            expect_res = [29.8317]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_SGDOptimizer_minimize_parameter_list():
    """
    test SGDOptimizer : parameter_list = ["fc.w_0"]
    default : startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
            optimizer.minimize(out, parameter_list=["fc.w_0"])
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.992, 0.992, 0.992],
                        [0.98800004, 0.98800004, 0.98800004]]
            expect_b = [0., 0., 0.]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_SGDOptimizer_minimize_no_grad_set():
    """
    test SGDOptimizer : no_grad_set = ["fc.w_0"]
    default : startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
            optimizer.minimize(out, no_grad_set=["fc.w_0"])
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[1., 1., 1.], [1., 1., 1.]]
            expect_b = [-0.004, -0.004, -0.004]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


#  AdagradOptimizer
def test_AdagradOptimizer():
    """
    test AdagradOptimizer
    default : learning_rate=0.2, epsilon=1e-06, regularization=None, name=None, initial_accumulator_value=0.0
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.2)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.65857875, 0.65857875, 0.65857875],
                        [0.65857863, 0.65857863, 0.65857863]]
            expect_b = [-0.34142125, -0.34142125, -0.34142125]
            expect_res = [22.800003]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_AdagradOptimizer_learning_rate():
    """
    test AdagradOptimizer : learning_rate_float=0.01
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.01)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.99, 0.99, 0.99], [0.99, 0.99, 0.99]]
            expect_b = [-0.01, -0.01, -0.01]
            expect_res = [30]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_AdagradOptimizer_learning_rate_var():
    """
    test AdagradOptimizer : learning_rate_variable=0.01
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            lr = fluid.layers.create_global_var(
                shape=[1],
                value=0.01,
                dtype='float32',
                persistable=True,
                name="lr")
            optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=lr)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.99, 0.99, 0.99], [0.99, 0.99, 0.99]]
            expect_b = [-0.01, -0.01, -0.01]
            expect_res = [30]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_AdagradOptimizer_epsilon():
    """
    test AdagradOptimizer : epsilon=0.0
    default : 1e-06
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.AdagradOptimizer(
                learning_rate=0.2, epsilon=0.0)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]]
            expect_b = [-0.2, -0.2, -0.2]
            expect_res = [30]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_AdagradOptimizer_regularization():
    """
    test AdagradOptimizer : regularization:fluid.regularizer.L2Decay(0.1)
    default : none
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.AdagradOptimizer(
                learning_rate=0.2,
                regularization=fluid.regularizer.L2Decay(
                    regularization_coeff=0.1))
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(
                    train_program,
                    feed={"inp": np_inp},
                    fetch_list=["fc.w_0@GRAD", "fc.b_0@GRAD", out.name])
            expect_w = [[4.08, 4.08, 4.08], [6.08, 6.08, 6.08]]
            expect_b = [1.98, 1.98, 1.98]
            expect_res = [22.800003]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_AdagradOptimizer_initial_accumulator_value():
    """
    test AdagradOptimizer : initial_accumulator_value=1.0
    default : 0.0
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.AdagradOptimizer(
                learning_rate=0.2, initial_accumulator_value=1.0, epsilon=0.0)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.8059715, 0.8059715, 0.8059715],
                        [0.8027212, 0.8027212, 0.8027212]]
            expect_b = [-0.17888544, -0.17888544, -0.17888544]
            expect_res = [30.]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_AdagradOptimizer_minimize_parameter_list():
    """
    test AdagradOptimizer : parameter_list = ["fc.w_0"]
    default : startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.2)
            optimizer.minimize(out, parameter_list=["fc.w_0"])
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.65857875, 0.65857875, 0.65857875],
                        [0.65857863, 0.65857863, 0.65857863]]
            expect_b = [0., 0., 0.]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_AdagradOptimizer_minimize_no_grad_set():
    """
    test AdagradOptimizer : no_grad_set = ["fc.w_0"]
    default : startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.2)
            optimizer.minimize(out, no_grad_set=["fc.w_0"])
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[1., 1., 1.], [1., 1., 1.]]
            expect_b = [-0.34142125, -0.34142125, -0.34142125]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


#  MomentumOptimizer
def test_MomentumOptimizer():
    """
    test MomentumOptimizer
    default : learning_rate=0.001, momentum=0.9, use_nesterov = false regularization = NONE 、 name = NONE
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            moment_optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001, momentum=0.9)
            moment_optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.9884, 0.9884, 0.9884],
                        [0.98260003, 0.98260003, 0.98260003]]
            expect_b = [-0.0058, -0.0058, -0.0058]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_MomentumOptimizer_lr_float():
    """
    test MomentumOptimizer : learning_rate_float=0.01, momentum=0.9
    :return :
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            moment_optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.01, momentum=0.9)
            moment_optimizer.minimize(out)
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.884, 0.884, 0.884], [0.826, 0.826, 0.826]]
            expect_b = [-0.058, -0.058, -0.058]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_MomentumOptimizer_lr_var():
    """
    test MomentumOptimizer : learning_rate_variable=0.01, momentum=0.9
    :return :
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            lr = fluid.layers.create_global_var(
                shape=[1],
                value=0.01,
                dtype='float32',
                persistable=True,
                name="lr")
            moment_optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=lr, momentum=0.9)
            moment_optimizer.minimize(out)
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.884, 0.884, 0.884], [0.826, 0.826, 0.826]]
            expect_b = [-0.058, -0.058, -0.058]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_MomentumOptimizer_momentum():
    """
    test MomentumOptimizer : momentum=0.0
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            moment_optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001, momentum=0.0)
            moment_optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.992, 0.992, 0.992],
                        [0.98800004, 0.98800004, 0.98800004]]
            expect_b = [-0.004, -0.004, -0.004]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_MomentumOptimizer_use_nesterov():
    """
    test MomentumOptimizer : use_nesterov = True
    :return: fc.w_0, fc.b_0
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            moment_optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001, momentum=0.9, use_nesterov=True)
            moment_optimizer.minimize(out)
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.98156, 0.98156, 0.98156],
                        [0.97234, 0.97234, 0.97234]]
            expect_b = [-0.00922, -0.00922, -0.00922]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_MomentumOptimizer_use_nesterov_momentum():
    """
    test MomentumOptimizer : use_nesterov = True, momentum=0.0
    :return: fc.w_0, fc.b_0
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            moment_optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001, momentum=0.0, use_nesterov=True)
            moment_optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.992, 0.992, 0.992],
                        [0.98800004, 0.98800004, 0.98800004]]
            expect_b = [-0.004, -0.004, -0.004]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_MomentumOptimizer_regularization():
    """
    test MomentumOptimizer : regularization:fluid.regularizer.L2Decay(0.01)
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001,
                momentum=0.9,
                regularization=fluid.regularizer.L2Decay(
                    regularization_coeff=0.01))
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(
                    train_program,
                    feed={"inp": np_inp},
                    fetch_list=["fc.w_0@GRAD", "fc.b_0@GRAD", out.name])
            expect_w = [[4.009959, 4.009959, 4.009959],
                        [6.009939, 6.009939, 6.009939]]
            expect_b = [1.99998, 1.99998, 1.99998]
            expect_res = [29.8317]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_MomentumOptimizer_minimize_parameter_list():
    """
    test MomentumOptimizer : parameter_list = ["fc.w_0"]
    default : startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001, momentum=0.9)
            optimizer.minimize(out, parameter_list=["fc.w_0"])
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.9884, 0.9884, 0.9884],
                        [0.98260003, 0.98260003, 0.98260003]]
            expect_b = [0., 0., 0.]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_MomentumOptimizer_minimize_no_grad_set():
    """
    test MomentumOptimizer : no_grad_set = ["fc.w_0"]
    default : startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001, momentum=0.9)
            optimizer.minimize(out, no_grad_set=["fc.w_0"])
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[1., 1., 1.], [1., 1., 1.]]
            expect_b = [-0.0058, -0.0058, -0.0058]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


#  DecayedAdagradOptimizer
def test_DecayedAdagradOptimizer():
    """
    test DecayedAdagradOptimizer
    default : lr = 0.2, decay=0.95, epsilon=1e-06, regularization=None, name=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.DecayedAdagradOptimizer(
                learning_rate=0.2)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[-0.53493816, -0.53493816, -0.53493816],
                        [-0.5349387, -0.5349387, -0.5349387]]
            expect_b = [-1.5349367, -1.5349367, -1.5349367]
            expect_res = [-2.1993403]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_DecayedAdagradOptimizer_learning_rate_float():
    """
    test DecayedAdagradOptimizer : learning_rate_float=0.01
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.DecayedAdagradOptimizer(
                learning_rate=0.01)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.9552787, 0.9552787, 0.9552787],
                        [0.9552787, 0.9552787, 0.9552787]]
            expect_b = [-0.04472125, -0.04472125, -0.04472125]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_DecayedAdagradOptimizer_learning_rate_var():
    """
    test DecayedAdagradOptimizer : learning_rate_variable=0.01
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            lr = fluid.layers.create_global_var(
                shape=[1],
                value=0.01,
                dtype='float32',
                persistable=True,
                name="lr")
            optimizer = fluid.optimizer.DecayedAdagradOptimizer(
                learning_rate=lr)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.9552787, 0.9552787, 0.9552787],
                        [0.9552787, 0.9552787, 0.9552787]]
            expect_b = [-0.04472125, -0.04472125, -0.04472125]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_DecayedAdagradOptimizer_decay():
    """
    test DecayedAdagradOptimizer : decay没有生效 http://newicafe.baidu.com/issue/DLTP-3183/show?cid=5
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.DecayedAdagradOptimizer(
                learning_rate=0.2, decay=0.6, epsilon=0.0)
            optimizer.minimize(out)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            for i in range(5):
                res = exe.run(feed={"inp": np_inp},
                              fetch_list=[
                                  "fc.w_0_moment_0", "fc.b_0_moment_0",
                                  "fc.w_0", "fc.b_0"
                              ])
                # print(res[0])


def test_DecayedAdagradOptimizer_epsilon():
    """
    test DecayedAdagradOptimizer : epsilon=0.0
    default : 1e-06
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.DecayedAdagradOptimizer(
                learning_rate=0.2, epsilon=0.0)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[-0.5349397, -0.5349397, -0.5349397],
                        [-0.5349397, -0.5349397, -0.5349397]]
            expect_b = [-1.5349398, -1.5349398, -1.5349398]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_DecayedAdagradOptimizer_regularization():
    """
    test DecayedAdagradOptimizer : regularization:fluid.regularizer.L2Decay(0.1)
    default : none
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            # out = fluid.layers.fc(inp, size=3)
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.DecayedAdagradOptimizer(
                learning_rate=0.2,
                regularization=fluid.regularizer.L2Decay(
                    regularization_coeff=0.1))
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(
                    train_program,
                    feed={"inp": np_inp},
                    fetch_list=["fc.w_0@GRAD", "fc.b_0@GRAD", out.name])
            expect_w = [[4.010557, 4.010557, 4.010557],
                        [6.010557, 6.010557, 6.010557]]
            expect_b = [1.9105575, 1.9105575, 1.9105575]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_DecayedAdagradOptimizer_minimize_parameter_list():
    """
    test DecayedAdagradOptimizer : parameter_list = ["fc.w_0"]
    default : startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.DecayedAdagradOptimizer(
                learning_rate=0.2)
            optimizer.minimize(out, parameter_list=["fc.w_0"])
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.10557389, 0.10557389, 0.10557389],
                        [0.10557353, 0.10557353, 0.10557353]]
            expect_b = [0., 0., 0.]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_DecayedAdagradOptimizer_minimize_no_grad_set():
    """
    test DecayedAdagradOptimizer : no_grad_set = ["fc.w_0"]
    default : startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.DecayedAdagradOptimizer(
                learning_rate=0.2)
            optimizer.minimize(out, no_grad_set=["fc.w_0"])
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[1., 1., 1.], [1., 1., 1.]]
            expect_b = [-0.8944251, -0.8944251, -0.8944251]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


#  RMSPropOptimizer
def test_RMSPropOptimizer():
    """
    test RMSPropOptimizer :
    default : learning_rate=0.1, rho=0.95, epsilon=1e-06, momentum=0.0, centered=False, regularization=None, name=None
    :return:
    rmsprop
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.RMSPropOptimizer(learning_rate=0.1)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.5527867, 0.5527867, 0.5527867],
                        [0.5527866, 0.5527866, 0.5527866]]
            expect_b = [-0.44721246, -0.44721246, -0.44721246]
            expect_res = [30.0]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_RMSPropOptimizer_learning_rate_float():
    """
    test RMSPropOptimizer : learning_rate=0.01
    default : 0.1
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.RMSPropOptimizer(learning_rate=0.01)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.9552787, 0.9552787, 0.9552787],
                        [0.95527864, 0.95527864, 0.95527864]]
            expect_b = [-0.04472124, -0.04472124, -0.04472124]
            expect_res = [30.0]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_RMSPropOptimizer_rho():
    """
    test RMSPropOptimizer : rho = 0.1
    default : 0.95
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.RMSPropOptimizer(
                learning_rate=0.1, rho=0.1, epsilon=0.0)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.89459074, 0.89459074, 0.89459074],
                        [0.89459074, 0.89459074, 0.89459074]]
            expect_b = [-0.10540926, -0.10540926, -0.10540926]
            expect_res = [30.0]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_RMSPropOptimizer_epsilon():
    """
    test RMSPropOptimizer : epsilon = 1.0
    default : 1e-6
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.RMSPropOptimizer(
                learning_rate=0.1, epsilon=1.0)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.70185757, 0.70185757, 0.70185757],
                        [0.64143145, 0.64143145, 0.64143145]]
            expect_b = [-0.18257418, -0.18257418, -0.18257418]
            expect_res = [30.0]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_RMSPropOptimizer_momentum():
    """
    test RMSPropOptimizer : momentum = 0.1
    default : 0.0
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.RMSPropOptimizer(
                learning_rate=0.1, momentum=1.0)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[-0.21468276, -0.21468276, -0.21468276],
                        [-0.21468306, -0.21468306, -0.21468306]]
            expect_b = [-1.2146808, -1.2146808, -1.2146808]
            expect_res = [13.900324]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_RMSPropOptimizer_centered():
    """
    test RMSPropOptimizer : centered = True
    default : False
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.RMSPropOptimizer(
                learning_rate=0.1, centered=True)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.54116887, 0.54116887, 0.54116887],
                        [0.5411687, 0.5411687, 0.5411687]]
            expect_b = [-0.4588302, -0.4588302, -0.4588302]
            expect_res = [30.]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_RMSPropOptimizer_regularization():
    """
    test RMSPropOptimizer : regularization_coeff = 0.1
    default : none
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.RMSPropOptimizer(
                learning_rate=0.1,
                regularization=fluid.regularizer.L2Decay(
                    regularization_coeff=0.1))
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(
                    train_program,
                    feed={"inp": np_inp},
                    fetch_list=["fc.w_0@GRAD", "fc.b_0@GRAD", out.name])
            expect_w = [[4.1, 4.1, 4.1], [6.1, 6.1, 6.1]]
            expect_b = [2., 2., 2.]
            expect_res = [30.]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_RMSPropOptimizer_minimize_parameter_list():
    """
    test RMSPropOptimizer : parameter_list = ["fc.w_0"]
    default : startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.RMSPropOptimizer(learning_rate=0.1)
            optimizer.minimize(out, parameter_list=["fc.w_0"])
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.5527867, 0.5527867, 0.5527867],
                        [0.5527866, 0.5527866, 0.5527866]]
            expect_b = [0., 0., 0.]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


def test_RMSPropOptimizer_minimize_no_grad_set():
    """
    test RMSPropOptimizer : no_grad_set = ["fc.w_0"]
    default : startup_program=None, parameter_list=None, no_grad_set=None, grad_clip=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.RMSPropOptimizer(learning_rate=0.1)
            optimizer.minimize(out, no_grad_set=["fc.w_0"])
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[1., 1., 1.], [1., 1., 1.]]
            expect_b = [-0.44721246, -0.44721246, -0.44721246]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)


#  ModelAverage
def test_ModelAverage():
    """
    test ModelAverage : average_window_rate = 1.0 , min_average_window=2, max_average_window=3
    regularization=None, name=None
    :return: fc.w_0, fc.b_0
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)

            moment_optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001, momentum=0.9)
            moment_optimizer.minimize(out)

            model_average = fluid.optimizer.ModelAverage(
                average_window_rate=1.0,
                min_average_window=2,
                max_average_window=3)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(3):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=[
                                  "fc.w_0",
                                  "fc.b_0",
                              ])
            expect_w_restore = res[0]
            expect_b_restore = res[1]

            inference_program = fluid.default_main_program().clone(
                for_test=True)
            with model_average.apply(exe, need_restore=True):
                res = exe.run(inference_program,
                              feed={"inp": np_inp},
                              fetch_list=[
                                  "fc.w_0",
                                  "fc.b_0",
                              ])
            expect_w_apply = [[0.98732, 0.98732, 0.98732],
                              [0.98098, 0.98098, 0.98098]]
            expect_b_apply = [-0.00634, -0.00634, -0.00634]
            tools.compare(res[0], expect_w_apply)
            tools.compare(res[1], expect_b_apply)

            model_average.restore(exe)
            res = exe.run(inference_program,
                          feed={"inp": np_inp},
                          fetch_list=[
                              "fc.w_0",
                              "fc.b_0",
                          ])
            tools.compare(res[0], expect_w_restore)
            tools.compare(res[1], expect_b_restore)


def test_ModelAverage_average_window_rate():
    """
    test ModelAverage : average_window_rate=0.1
     min_average_window=3, max_average_window=3,
     regularization=None, name=None
    :return: fc.w_0, fc.b_0
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)

            moment_optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001, momentum=0.9)
            moment_optimizer.minimize(out)

            model_average = fluid.optimizer.ModelAverage(
                average_window_rate=0.1,
                min_average_window=3,
                max_average_window=3)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(3):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=[
                                  "fc.w_0",
                                  "fc.b_0",
                              ])
            expect_w_restore = res[0]
            expect_b_restore = res[1]
            #
            inference_program = fluid.default_main_program().clone(
                for_test=True)
            with model_average.apply(exe, need_restore=True):
                res = exe.run(inference_program,
                              feed={"inp": np_inp},
                              fetch_list=[
                                  "fc.w_0",
                                  "fc.b_0",
                              ])
            expect_w_apply = [[0.98732, 0.98732, 0.98732],
                              [0.98098, 0.98098, 0.98098]]
            expect_b_apply = [-0.00634, -0.00634, -0.00634]
            tools.compare(res[0], expect_w_apply)
            tools.compare(res[1], expect_b_apply)

            model_average.restore(exe)
            res = exe.run(inference_program,
                          feed={"inp": np_inp},
                          fetch_list=[
                              "fc.w_0",
                              "fc.b_0",
                          ])
            tools.compare(res[0], expect_w_restore)
            tools.compare(res[1], expect_b_restore)


def test_ModelAverage_need_restore():
    """
    test ModelAverage : need_restore写死成True了 http://newicafe.baidu.com/issue/DLTP-3279/show?cid=5
    :return: fc.w_0, fc.b_0
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)

            moment_optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001, momentum=0.9)
            moment_optimizer.minimize(out)

            model_average = fluid.optimizer.ModelAverage(
                average_window_rate=1.0,
                min_average_window=2,
                max_average_window=3)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            for i in range(3):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=[
                                  "fc.w_0",
                                  "fc.b_0",
                              ])
            expect_w_restore = res[0]
            expect_b_restore = res[1]

            inference_program = fluid.default_main_program().clone(
                for_test=True)
            with model_average.apply(exe, need_restore=False):
                res = exe.run(inference_program,
                              feed={"inp": np_inp},
                              fetch_list=[
                                  "fc.w_0",
                                  "fc.b_0",
                              ])
            expect_w_apply = [[0.98732, 0.98732, 0.98732],
                              [0.98098, 0.98098, 0.98098]]
            expect_b_apply = [-0.00634, -0.00634, -0.00634]
            tools.compare(res[0], expect_w_apply)
            tools.compare(res[1], expect_b_apply)

            model_average.restore(exe)
            res = exe.run(inference_program,
                          feed={"inp": np_inp},
                          fetch_list=[
                              "fc.w_0",
                              "fc.b_0",
                          ])
            tools.compare(res[0], expect_w_restore)
            tools.compare(res[1], expect_b_restore)


#  ExponentialMovingAverage
def test_ExponentialMovingAverage():
    """
    test ExponentialMovingAverage : 偏置校正计算与官网不一致 http://newicafe.baidu.com/issue/DLTP-3276/show?cid=5
    :return: fc.w_0, fc.b_0
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            test_program = fluid.default_main_program().clone(for_test=True)

            moment_optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001, momentum=0.9)
            moment_optimizer.minimize(out)

            EMA = fluid.optimizer.ExponentialMovingAverage(decay=0.999)
            EMA.update()

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            for i in range(3):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=[
                                  "fc.w_0",
                                  "fc.b_0",
                              ])
                # print(res)
            # expect_w_restore = [[0.97756, 0.97756, 0.97756], [0.96634, 0.96634, 0.96634]]
            # expect_b_restore = [-0.01122, -0.01122, -0.01122]
            with EMA.apply(exe, need_restore=True):
                res = exe.run(
                    test_program,
                    feed={"inp": np_inp},
                    fetch_list=[
                        # "learning_rate_0",
                        # "scheduled_ema_decay_rate",
                        # "fc.w_0.ema_tmp_0",
                        # "fc.b_0.ema_tmp_0",
                        "fc.w_0_ema_0",
                        "fc.b_0_ema_0",
                        "fc.w_0",
                        "fc.b_0",
                    ])
                # print(res)
                # print()
            # expect_w_apply =
            # expect_b_apply =
            # tools.compare(res[0], expect_w_apply)
            # tools.compare(res[1], expect_b_apply)

            # EMA.restore(exe)
            # res = exe.run(test_program, feed={"inp": np_inp}, fetch_list=[
            #     "fc.w_0",
            #     "fc.b_0",
            # ])
            # tools.compare(res[0], expect_w_restore)
            # tools.compare(res[1], expect_b_restore)


def test_ExponentialMovingAverage_thres_steps():
    """
    test ExponentialMovingAverage : 偏置校正计算与官网不一致 http://newicafe.baidu.com/issue/DLTP-3276/show?cid=5
    (偏置校正计算不一致、无法正确更新参数、无法验证thres_steps是否生效)
    thres_steps = 1
    :return: fc.w_0, fc.b_0
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            test_program = fluid.default_main_program().clone(for_test=True)

            moment_optimizer = fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001, momentum=0.9)
            moment_optimizer.minimize(out)
            ts = fluid.layers.create_global_var(
                shape=[1],
                value=1.0,
                dtype='float32',
                persistable=True,
                name="ts")
            EMA = fluid.optimizer.ExponentialMovingAverage(
                decay=0.999, thres_steps=ts)
            EMA.update()

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=[
                                  "fc.w_0",
                                  "fc.b_0",
                              ])
                # print(res)
            # expect_w_restore = res[0]
            # expect_b_restore = res[1]
            with EMA.apply(exe, need_restore=True):
                res = exe.run(
                    test_program,
                    feed={"inp": np_inp},
                    fetch_list=[
                        # "learning_rate_0",
                        # "scheduled_ema_decay_rate",
                        # "fc.w_0.ema_tmp_0",
                        # "fc.b_0.ema_tmp_0",
                        "fc.w_0_ema_0",
                        "fc.b_0_ema_0",
                        "fc.w_0",
                        "fc.b_0",
                    ])
                # print()
                # print(res)
            # expect_w_apply =
            # expect_b_apply =
            # tools.compare(res[0], expect_w_apply)
            # tools.compare(res[1], expect_b_apply)

            # EMA.restore(exe)
            # res = exe.run(test_program, feed={"inp": np_inp}, fetch_list=[
            #     "fc.w_0",
            #     "fc.b_0",
            # ])
            # tools.compare(res[0], expect_w_restore)
            # tools.compare(res[1], expect_b_restore)


            #  LarsMomentumOptimizer
def test_LarsMomentumOptimizer():
    """
    test LarsMomentumOptimizer
    default : learning_rate, momentum, lars_coeff=0.001, lars_weight_decay=0.0005, regularization=None, name=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.LarsMomentumOptimizer(
                learning_rate=0.001, momentum=0.9)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.99999774, 0.99999774, 0.99999774],
                        [0.99999654, 0.99999654, 0.99999654]]
            expect_b = [-0.0038, -0.0038, -0.0038]
            expect_res = [29.98797]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_LarsMomentumOptimizer_learning_rate_float():
    """
    test LarsMomentumOptimizer learning_rate=0.1
    default :
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.LarsMomentumOptimizer(
                learning_rate=0.1, momentum=0.9)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.99977252, 0.99977252, 0.99977252],
                        [0.99965878, 0.99965878, 0.99965878]]
            expect_b = [-0.38002, -0.38002, -0.38002]
            expect_res = [28.79694]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_LarsMomentumOptimizer_moment():
    """
    test LarsMomentumOptimizer : momentum=0.1
    default :
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.LarsMomentumOptimizer(
                learning_rate=0.1, momentum=0.1)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=[
                                  "fc.w_0",
                                  "fc.b_0",
                                  out.name,
                              ])
            expect_w = [[0.99983525, 0.99983525, 0.99983525],
                        [0.99975294, 0.99975294, 0.99975294]]
            expect_b = [-0.22002, -0.22002, -0.22002]
            expect_res = [28.79694]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_LarsMomentumOptimizer_lars_coeff():
    """
    test LarsMomentumOptimizer : lars_coeff=0.01
    default :
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.LarsMomentumOptimizer(
                learning_rate=0.1, momentum=0.9, lars_coeff=0.1)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
                # print(res)
                # print()
            expect_w = [[0.9773268, 0.9773268, 0.9773268],
                        [0.9659917, 0.9659917, 0.9659917]]
            expect_b = [-0.3819998, -0.3819998, -0.3819998]
            expect_res = [28.49406]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_LarsMomentumOptimizer_lars_weight_decay():
    """
    test LarsMomentumOptimizer : lars_weight_decay=0.1
    default :
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.LarsMomentumOptimizer(
                learning_rate=0.1, momentum=0.9, lars_weight_decay=0.1)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(2):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=["fc.w_0", "fc.b_0", out.name])
            expect_w = [[0.9997713, 0.9997713, 0.9997713],
                        [0.9996598, 0.9996598, 0.9996598]]
            expect_b = [-0.3800196, -0.3800196, -0.3800196]
            expect_res = [28.796944]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


#  LookaheadOptimizer
def test_LookaheadOptimizer():
    """
    test LookaheadOptimizer :
    default : inner_optimizer, alpha=0.5, k=5
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            sgd = fluid.optimizer.SGD(learning_rate=0.01)
            optimizer = fluid.optimizer.LookaheadOptimizer(sgd, alpha=0.5, k=5)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(5):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=[
                                  "fc.w_0@SLOW", "fc.b_0@SLOW", "fc.w_0",
                                  "fc.b_0", out.name
                              ])
            expect_w_slow = [[0.9, 0.9, 0.9], [0.85, 0.85, 0.85]]
            expect_b_slow = [-0.05, -0.05, -0.05]
            expect_w_fast = [[0.9, 0.9, 0.9], [0.85, 0.85, 0.85]]
            expect_b_fats = [-0.05, -0.05, -0.05]
            expect_res = [23.279999]
            tools.compare(res[0], expect_w_slow)
            tools.compare(res[1], expect_b_slow)
            tools.compare(res[2], expect_w_fast)
            tools.compare(res[3], expect_b_fats)
            tools.compare(res[4], expect_res)


def test_LookaheadOptimizer_adagrad():
    """
    test LookaheadOptimizer : inner_optimizer = AdagradOptimizer
    default : inner_optimizer, alpha=0.5, k=5
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            adagrad = fluid.optimizer.AdagradOptimizer(learning_rate=0.2)
            optimizer = fluid.optimizer.LookaheadOptimizer(
                adagrad, alpha=0.5, k=5)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(5):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=[
                                  "fc.w_0@SLOW", "fc.b_0@SLOW", "fc.w_0",
                                  "fc.b_0", out.name
                              ])
            expect_w_slow = [[0.67683303, 0.67683303, 0.67683303],
                             [0.67683303, 0.67683303, 0.67683303]]
            expect_b_slow = [-0.32316697, -0.32316697, -0.32316697]
            expect_w_fast = [[0.67683303, 0.67683303, 0.67683303],
                             [0.67683303, 0.67683303, 0.67683303]]
            expect_b_fats = [-0.32316697, -0.32316697, -0.32316697]
            expect_res = [9.951912]
            tools.compare(res[0], expect_w_slow)
            tools.compare(res[1], expect_b_slow)
            tools.compare(res[2], expect_w_fast)
            tools.compare(res[3], expect_b_fats)
            tools.compare(res[4], expect_res)


def test_LookaheadOptimizer_alpha():
    """
    test LookaheadOptimizer : alpha =0.1
    default : inner_optimizer, alpha=0.5, k=5
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            sgd = fluid.optimizer.SGD(learning_rate=0.01)
            optimizer = fluid.optimizer.LookaheadOptimizer(sgd, alpha=1.0, k=5)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(5):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=[
                                  "fc.w_0@SLOW", "fc.b_0@SLOW", "fc.w_0",
                                  "fc.b_0", out.name
                              ])
            expect_w_slow = [[0.8, 0.8, 0.8], [0.7, 0.7, 0.7]]
            expect_b_slow = [-0.1, -0.1, -0.1]
            expect_w_fast = [[0.8, 0.8, 0.8], [0.7, 0.7, 0.7]]
            expect_b_fats = [-0.1, -0.1, -0.1]
            expect_res = [23.279999]
            tools.compare(res[0], expect_w_slow)
            tools.compare(res[1], expect_b_slow)
            tools.compare(res[2], expect_w_fast)
            tools.compare(res[3], expect_b_fats)
            tools.compare(res[4], expect_res)


def test_LookaheadOptimizer_k():
    """
    test LookaheadOptimizer : k = 10
    default : inner_optimizer, alpha=0.5, k=5
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            sgd = fluid.optimizer.SGD(learning_rate=0.01)
            optimizer = fluid.optimizer.LookaheadOptimizer(sgd, alpha=0.5, k=10)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(10):
                res = exe.run(train_program,
                              feed={"inp": np_inp},
                              fetch_list=[
                                  "fc.w_0@SLOW", "fc.b_0@SLOW", "fc.w_0",
                                  "fc.b_0", out.name
                              ])
            expect_w_slow = [[0.8, 0.8, 0.8], [0.7, 0.7, 0.7]]
            expect_b_slow = [-0.1, -0.1, -0.1]
            expect_w_fast = [[0.8, 0.8, 0.8], [0.7, 0.7, 0.7]]
            expect_b_fats = [-0.1, -0.1, -0.1]
            expect_res = [14.879996]
            tools.compare(res[0], expect_w_slow)
            tools.compare(res[1], expect_b_slow)
            tools.compare(res[2], expect_w_fast)
            tools.compare(res[3], expect_b_fats)
            tools.compare(res[4], expect_res)


# AdamOptimizer
def test_AdamOptimizer():
    """
    test AdamOptimizer
    default : http://newicafe.baidu.com/issue/DLTP-3196/show 返回不一致
    lr = 0.1 、beta1 = 0.9、 beta2 = 0.999、 epsilon = 1e-08 、 regularization = NONE 、 name = NONE 、lazy_mode = False
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.1)
            optimizer.minimize(out)

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(
                    train_program,
                    feed={"inp": np_inp},
                    fetch_list=[
                        "learning_rate_0",  # 0
                        "fc.w_0@GRAD",  # 1
                        "fc.b_0@GRAD",  # 2
                        "fc.w_0_beta1_pow_acc_0",  # 3
                        "fc.w_0_beta2_pow_acc_0",  # 4
                        "fc.b_0_beta1_pow_acc_0",  # 5
                        "fc.b_0_beta2_pow_acc_0",  # 6
                        "fc.w_0_moment1_0",  # 7
                        "fc.w_0_moment2_0",  # 8
                        "fc.b_0_moment1_0",  # 9
                        "fc.b_0_moment2_0",  # 10
                        "fc.w_0",  # 11
                        "fc.b_0",  # 12
                        out.name
                    ])
            # expect_w = [[0.92558715, 0.92558715, 0.92558715], [0.92558696, 0.92558696, 0.92558696]]
            # expect_b = [-0.07441226, -0.07441226, -0.07441226]
            # expect_res = [30.0]
            # tools.compare(res[0], expect_w)
            # tools.compare(res[1], expect_b)
            # tools.compare(res[4], expect_res)


            # AdamaxOptimizer
def test_AdamaxOptimizer():
    """
    test AdamaxOptimizer : http://newicafe.baidu.com/issue/DLTP-3198/show?cid=5
    default : lr = 0.1 、beta1 = 0.9、 beta2 = 0.999、 epsilon = 1e-08 、 regularization = NONE 、 name = NONE 、
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.AdamaxOptimizer(
                learning_rate=0.001, beta1=0.1, beta2=0.9, epsilon=0.0)
            optimizer.minimize(out)

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(
                    train_program,
                    feed={"inp": np_inp},
                    fetch_list=[
                        "learning_rate_0",  # 0
                        "fc.w_0@GRAD",  # 1
                        "fc.b_0@GRAD",  # 2
                        "fc.w_0_beta1_pow_acc_0",  # 3
                        "fc.b_0_beta1_pow_acc_0",  # 4
                        "fc.w_0_moment_0",  # 5
                        "fc.b_0_moment_0",  # 6
                        "fc.w_0_inf_norm_0",  # 7
                        "fc.b_0_inf_norm_0",  # 8
                        "fc.w_0",  # 9
                        "fc.b_0",  # 10
                        out.name
                    ])
                # print(res[9], res[10])


            # AdadeltaOptimizer
def test_AdadeltaOptimizer():
    """
    test AdadeltaOptimizer : http://newicafe.baidu.com/issue/DLTP-3199/show?cid=5
    default : lr = 0.01 、 regularization = NONE 、 name = NONE 、
    lazy_mode = False
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.AdadeltaOptimizer(learning_rate=0.01, )
            optimizer.minimize(out)

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(1):
                res = exe.run(
                    train_program,
                    feed={"inp": np_inp},
                    fetch_list=[
                        "learning_rate_0",  # 0
                        "fc.w_0@GRAD",  # 1
                        "fc.b_0@GRAD",  # 2
                        "fc.w_0__avg_squared_update_0",  # 3
                        "fc.b_0__avg_squared_update_0",  # 4
                        "fc.w_0__avg_squared_grad_0",  # 5
                        "fc.b_0__avg_squared_grad_0",  # 6
                        "fc.w_0",  # 7
                        "fc.b_0",  # 8
                        out.name  # 9
                    ])
            # print(res[7], res[8])


            #  LambOptimizer
def test_LambOptimizer():
    """
    test LambOptimizer ：
    default :learning_rate=0.001, lamb_weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-06, regularization=None,
    exclude_from_weight_decay_fn=None, name=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.LambOptimizer(learning_rate=0.001)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            for i in range(2):
                res = exe.run(
                    train_program,
                    feed={"inp": np_inp},
                    fetch_list=[
                        # "fc.w_0_moment1_0",
                        # "fc.b_0_moment1_0",
                        #
                        # "fc.w_0_moment2_0",
                        # "fc.b_0_moment2_0",
                        "fc.w_0",
                        "fc.b_0",
                        out.name
                    ])
            expect_w = [[0.998001, 0.998001, 0.998001],
                        [0.998001, 0.998001, 0.998001]]
            expect_b = [-0.00316541, -0.00316541, -0.00316541]
            expect_res = [29.951027]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)


def test_LambOptimizer_learning_rate_float():
    """
    test LambOptimizer ：learning_rate = 0.1
    default
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(name="fc",
                                  input=inp,
                                  size=3,
                                  param_attr=fluid.initializer.Constant(
                                      value=1.0, force_cpu=True))
            out = fluid.layers.reduce_sum(out)

            optimizer = fluid.optimizer.LambOptimizer(learning_rate=0.1)
            optimizer.minimize(out)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            for i in range(2):
                res = exe.run(
                    train_program,
                    feed={"inp": np_inp},
                    fetch_list=[
                        # "fc.w_0_moment1_0",
                        # "fc.b_0_moment1_0",
                        #
                        # "fc.w_0_moment2_0",
                        # "fc.b_0_moment2_0",
                        "fc.w_0",
                        "fc.b_0",
                        out.name
                    ])
            expect_w = [[0.81000024, 0.81000024, 0.81000024],
                        [0.80999977, 0.80999977, 0.80999977]]
            expect_b = [-0.34784739, -0.34784739, -0.34784739]
            expect_res = [25.10265]
            tools.compare(res[0], expect_w)
            tools.compare(res[1], expect_b)
            tools.compare(res[2], expect_res)
