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
"""test static paramattr."""

import paddle.fluid as fluid
import numpy
import tools
import math
use_cuda = False



def test_constantInitializer():
    """
    test constant Initializer
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.ConstantInitializer(value=2.0)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res = exe.run(compiled_prog,
                             feed={"X": x},
                             fetch_list=["fc.w_0"])[0][0]
        expect = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        tools.compare(res, expect)


def test_constantInitializerAlias():
    """
    test constant Initializer alias
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.Constant(value=2.0)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res = exe.run(compiled_prog,
                             feed={"X": x},
                             fetch_list=["fc.w_0"])[0][0]
        expect = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        tools.compare(res, expect)


def test_constantInitializer1():
    """
    test constant Initializer , force_cpu=True
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.ConstantInitializer(value=2.0, force_cpu=True)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res = exe.run(compiled_prog,
                             feed={"X": x},
                             fetch_list=["fc.w_0"])[0][0]
        expect = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        tools.compare(res, expect)


def test_uniformInitializer():
    """
    test Uniform Initializer
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.UniformInitializer(low=-1.0, high=1.0, seed=33)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.UniformInitializer(low=-1.0, high=1.0, seed=33)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res1 = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
        # expect = [-0.99851644, 0.6121416,  0.6892719, -0.15365571, 0.8850763, -0.48047507,
        #           0.9884044, -0.7314464, 0.35180688, 0.07182181]
        tools.compare(res, res1)
        print("[Result] ====> {}".format(res))


def test_uniformInitializerAlias():
    """
    test Uniform Initializer
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.Uniform(low=-1.0, high=1.0, seed=33)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.UniformInitializer(low=-1.0, high=1.0, seed=33)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res1 = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
        # expect = [-0.99851644, 0.6121416,  0.6892719, -0.15365571, 0.8850763, -0.48047507,
        #           0.9884044, -0.7314464, 0.35180688, 0.07182181]
        tools.compare(res, res1)
        print("[Result] ====> {}".format(res))


def test_uniformInitializer1():
    """
    test Uniform Initializer , low and high != 1.0
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.UniformInitializer(low=-5.0, high=5.0, seed=33)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.UniformInitializer(low=-5.0, high=5.0, seed=33)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res1 = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
        # expect = [-4.9925823, 3.060708, 3.4463596, -0.7682786, 4.4253817, -2.4023752,
        #           4.9420223, -3.6572318, 1.7590342, 0.35910892]
        tools.compare(res, res1)
        print("[Result] ====> {}".format(res))


def test_uniformInitializer2():
    """
    test Uniform Initializer , change seed
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.UniformInitializer(low=-1.0, high=1.0, seed=66)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.UniformInitializer(low=-1.0, high=1.0, seed=66)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res1 = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
        # expect = [-0.99703294, 0.22428334, 0.37854385, 0.6926886, 0.7701527, 0.03904986,
        #           0.9768088, -0.46289277, -0.29638618, -0.8563564]
        tools.compare(res, res1)
        print("[Result] ====> {}".format(res))


def test_normalInitializer():
    """
    test Normal Initializer
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=33)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=33)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res1 = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
        # expect = [-0.25666377, 1.1513476, 0.40487382, 1.9832053, -0.00961026, 0.6131783,
        #           0.6937958, -0.92126787, 1.0037242, 0.08652732]
        tools.compare(res, res1)
        print("[Result] ====> {}".format(res))


def test_normalInitializerAlias():
    """
    test Normal Initializer alias
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.Normal(loc=0.0, scale=1.0, seed=33)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.Normal(loc=0.0, scale=1.0, seed=33)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res1 = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
        # expect = [-0.25666377, 1.1513476, 0.40487382, 1.9832053, -0.00961026, 0.6131783,
        #           0.6937958, -0.92126787, 1.0037242, 0.08652732]
        tools.compare(res, res1)
        print("[Result] ====> {}".format(res))


def test_normalInitializer1():
    """
    test Normal Initializer, loc = 5.0
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.NormalInitializer(loc=5.0, scale=1.0, seed=33)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.NormalInitializer(loc=5.0, scale=1.0, seed=33)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res1 = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
        # expect = [4.743336, 6.1513476, 5.404874, 6.9832053, 4.99039, 5.6131783,
        #           5.6937957, 4.078732, 6.003724, 5.0865273]
        tools.compare(res, res1)
        print("[Result] ====> {}".format(res))


def test_normalInitializer2():
    """
    test Normal Initializer, loc = 5.0 scale = 5.0
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.NormalInitializer(loc=5.0, scale=5.0, seed=33)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.NormalInitializer(loc=5.0, scale=5.0, seed=33)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res1 = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
        # expect = [3.716681, 10.756739, 7.0243692, 14.916027, 4.9519486, 8.065891,
        #           8.468979, 0.39366055, 10.018621, 5.4326367]
        tools.compare(res, res1)
        print("[Result] ====> {}".format(res))


def test_normalInitializer3():
    """
    test Normal Initializer, loc = 5.0 scale = 5.0 seed = 66
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.NormalInitializer(loc=5.0, scale=5.0, seed=66)
            param_attrs = fluid.ParamAttr(initializer=initializer)

            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            initializer = fluid.initializer.NormalInitializer(loc=5.0, scale=5.0, seed=66)
            param_attrs = fluid.ParamAttr(initializer=initializer)

            y_predict = fluid.layers.fc(name="fc", input=data, size=10, param_attr=param_attrs)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        numpy.random.seed(11)
        x = numpy.random.random(size=(10, 1)).astype('float32')
        compiled_prog = fluid.compiler.CompiledProgram(train_program)
        res1 = exe.run(compiled_prog,
                      feed={"X": x},
                      fetch_list=["fc.w_0"])[0][0]
        # expect = [9.267591, 7.3321743, 5.258151, 10.091329, 2.0341442, 3.973513,
        #           -3.696248, -4.2336454, 7.097207, 9.274171]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_BilinearInitializer():
    """
    test BilinearInitializer
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            factor = 2
            C = 2
            H = W = 32
            w_attr = fluid.ParamAttr(
                learning_rate=0.,
                regularizer=fluid.regularizer.L2Decay(0.),
                initializer=fluid.initializer.BilinearInitializer())
            x = fluid.layers.data(name="data", shape=[4, H, W],
                                  dtype="float32")
            conv_up = fluid.layers.conv2d_transpose(
                name="conv2d",
                input=x,
                num_filters=C,
                output_size=None,
                filter_size=2 * factor - factor % 2,
                padding=int(math.ceil((factor - 1) / 2.)),
                stride=factor,
                groups=C,
                param_attr=w_attr,
                bias_attr=False)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[1, 4, H, W]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog,
                           feed={"data": data},
                           fetch_list=["conv2d.w_0"])[0][0]
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            factor = 2
            C = 2
            H = W = 32
            w_attr = fluid.ParamAttr(
                learning_rate=0.,
                regularizer=fluid.regularizer.L2Decay(0.),
                initializer=fluid.initializer.BilinearInitializer())
            x = fluid.layers.data(name="data", shape=[4, H, W],
                                  dtype="float32")
            conv_up = fluid.layers.conv2d_transpose(
                name="conv2d",
                input=x,
                num_filters=C,
                output_size=None,
                filter_size=2 * factor - factor % 2,
                padding=int(math.ceil((factor - 1) / 2.)),
                stride=factor,
                groups=C,
                param_attr=w_attr,
                bias_attr=False)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[1, 4, H, W]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res1 = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["conv2d.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_BilinearInitializerAlias():
    """
    test BilinearInitializer  Alias
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            factor = 2
            C = 2
            H = W = 32
            w_attr = fluid.ParamAttr(
                learning_rate=0.,
                regularizer=fluid.regularizer.L2Decay(0.),
                initializer=fluid.initializer.Bilinear())
            x = fluid.layers.data(name="data", shape=[4, H, W],
                                  dtype="float32")
            conv_up = fluid.layers.conv2d_transpose(
                name="conv2d",
                input=x,
                num_filters=C,
                output_size=None,
                filter_size=2 * factor - factor % 2,
                padding=int(math.ceil((factor - 1) / 2.)),
                stride=factor,
                groups=C,
                param_attr=w_attr,
                bias_attr=False)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[1, 4, H, W]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog,
                           feed={"data": data},
                           fetch_list=["conv2d.w_0"])[0][0]
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            factor = 2
            C = 2
            H = W = 32
            w_attr = fluid.ParamAttr(
                learning_rate=0.,
                regularizer=fluid.regularizer.L2Decay(0.),
                initializer=fluid.initializer.Bilinear())
            x = fluid.layers.data(name="data", shape=[4, H, W],
                                  dtype="float32")
            conv_up = fluid.layers.conv2d_transpose(
                name="conv2d",
                input=x,
                num_filters=C,
                output_size=None,
                filter_size=2 * factor - factor % 2,
                padding=int(math.ceil((factor - 1) / 2.)),
                stride=factor,
                groups=C,
                param_attr=w_attr,
                bias_attr=False)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[1, 4, H, W]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res1 = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["conv2d.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_MSRAInitializer():
    """
    test MSRAInitializer
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)

    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            param_attr = fluid.initializer.MSRAInitializer(uniform=False)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 32, 32]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            param_attr = fluid.initializer.MSRAInitializer(uniform=False)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 32, 32]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res1 = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_MSRAInitializer1():
    """
    test MSRAInitializer with uniform = True
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)

    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            param_attr = fluid.initializer.MSRAInitializer(uniform=True)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 32, 32]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            param_attr = fluid.initializer.MSRAInitializer(uniform=True)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 32, 32]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res1 = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_MSRAInitializer2():
    """
    test MSRAInitializer with fan_in = float or string
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)

    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            param_attr = fluid.initializer.MSRAInitializer(uniform=False, fan_in="3.3")
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 32, 32]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            param_attr = fluid.initializer.MSRAInitializer(uniform=False, fan_in=3.3)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 32, 32]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res1 = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_MSRAInitializer3():
    """
    test MSRAInitializer with seed = 33
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)

    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            param_attr = fluid.initializer.MSRAInitializer(uniform=False, fan_in="3.3", seed=33)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 32, 32]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            param_attr = fluid.initializer.MSRAInitializer(uniform=False, fan_in=3.3, seed=33)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 32, 32]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res1 = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_MSRAInitializerAlias():
    """
    test MSRAInitializerAlias  MSRA
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)

    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            param_attr = fluid.initializer.MSRA(uniform=False, fan_in="3.3", seed=33)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 32, 32]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            param_attr = fluid.initializer.MSRA(uniform=False, fan_in=3.3, seed=33)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 32, 32]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res1 = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_NumpyArrayInitializer():
    """
    test NumpyArrayInitializer
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name="x", shape=[5], dtype='float32')
            param_attr = fluid.initializer.NumpyArrayInitializer(numpy.ones(shape=[5, 10]))
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 5]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res = exe.run(compiled_prog,
                           feed={"x": data},
                           fetch_list=["fc.w_0"])[0][0]
            expect = numpy.ones(shape=[10])
            tools.compare(res, expect)
            print("[Result] ====> {}".format(res))


def test_TruncatedNormalInitializer():
    """
    test TruncatedNormalInitializer
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)

    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.TruncatedNormalInitializer(loc=0.0, scale=2.0)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.TruncatedNormalInitializer(loc=0.0, scale=2.0)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res1 = exe.run(compiled_prog,
                           feed={"data": data},
                           fetch_list=["fc.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_TruncatedNormalInitializer1():
    """
    test TruncatedNormalInitializer loc = 1 scale = 4
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)

    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.TruncatedNormalInitializer(loc=1, scale=4)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.TruncatedNormalInitializer(loc=1, scale=4)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res1 = exe.run(compiled_prog,
                           feed={"data": data},
                           fetch_list=["fc.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_TruncatedNormalInitializer2():
    """
    test TruncatedNormalInitializer loc = 1 scale = 4 seed = 33
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)

    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.TruncatedNormalInitializer(loc=0, scale=2, seed=66)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.TruncatedNormalInitializer(loc=0, scale=2, seed=66)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res1 = exe.run(compiled_prog,
                           feed={"data": data},
                           fetch_list=["fc.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_TruncatedNormalInitializerAlias():
    """
    test TruncatedNormalInitializerAlias loc = 1 scale = 4 seed = 33
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)

    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.TruncatedNormal(loc=0, scale=2, seed=66)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.TruncatedNormal(loc=0, scale=2, seed=66)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res1 = exe.run(compiled_prog,
                           feed={"data": data},
                           fetch_list=["fc.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_XavierInitializer():
    """
    test XavierInitializer uniform=True
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)

    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.XavierInitializer()
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.XavierInitializer()
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res1 = exe.run(compiled_prog,
                           feed={"data": data},
                           fetch_list=["fc.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_XavierInitializer1():
    """
    test XavierInitializer uniform=False
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)

    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.XavierInitializer(uniform=False)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.XavierInitializer(uniform=False)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res1 = exe.run(compiled_prog,
                           feed={"data": data},
                           fetch_list=["fc.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_XavierInitializer2():
    """
    test XavierInitializer uniform=False seed = 66
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)

    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.XavierInitializer(uniform=False, seed=66)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.XavierInitializer(uniform=False, seed=66)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res1 = exe.run(compiled_prog,
                           feed={"data": data},
                           fetch_list=["fc.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_XavierInitializer3():
    """
    test XavierInitializer uniform=True fan_in != None fan_out != None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)

    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.XavierInitializer(fan_in=1, fan_out=10)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.XavierInitializer(fan_in=1, fan_out=10)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res1 = exe.run(compiled_prog,
                           feed={"data": data},
                           fetch_list=["fc.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))


def test_XavierInitializerAlias():
    """
    test XavierInitializerAlias uniform=True fan_in != None fan_out != None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)

    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.Xavier(fan_in=1, fan_out=10)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog,
                          feed={"data": data},
                          fetch_list=["fc.w_0"])[0][0]

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    numpy.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='data', shape=[1], dtype='float32')
            param_attr = fluid.initializer.Xavier(fan_in=1, fan_out=10)
            fc = fluid.layers.fc(name="fc", input=x, size=10, param_attr=param_attr)
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            data = numpy.ones(shape=[10, 1]).astype(numpy.float32)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)

            res1 = exe.run(compiled_prog,
                           feed={"data": data},
                           fetch_list=["fc.w_0"])[0][0]
    tools.compare(res, res1)
    print("[Result] ====> {}".format(res))
    
    
    
