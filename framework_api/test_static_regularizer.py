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
"""test static regularizer."""
import paddle.fluid as fluid
import numpy as np
import tools


def test_L1Decay():
    """
    test L1Decay
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            initializer = fluid.initializer.Constant(value=0.5)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc",
                                        input=data,
                                        size=10,
                                        param_attr=param_attrs)
            loss = fluid.layers.cross_entropy(input=y_predict, label=label)
            avg_loss = fluid.layers.mean(loss)
            optimizer = fluid.optimizer.Adagrad(
                learning_rate=0.1,
                regularization=fluid.regularizer.L1Decay(
                    regularization_coeff=0.1))
            optimizer.minimize(avg_loss)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            x = np.ones(shape=(10, 1)).astype('float32')
            y = np.zeros(shape=(10, 1)).astype('int64')
            for i in range(10):
                res = exe.run(compiled_prog,
                              feed={"X": x,
                                    "label": y},
                              fetch_list=[avg_loss])[0][0]
            res1 = res

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            initializer = fluid.initializer.Constant(value=0.5)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc",
                                        input=data,
                                        size=10,
                                        param_attr=param_attrs)
            loss = fluid.layers.cross_entropy(input=y_predict, label=label)
            avg_loss = fluid.layers.mean(loss)
            optimizer = fluid.optimizer.Adagrad(learning_rate=0.1, )
            optimizer.minimize(avg_loss)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            x = np.ones(shape=(10, 1)).astype('float32')
            y = np.zeros(shape=(10, 1)).astype('int64')
            for i in range(10):
                res = exe.run(compiled_prog,
                              feed={"X": x,
                                    "label": y},
                              fetch_list=[avg_loss])[0][0]
            res2 = res
    print(res1)
    print(res2)
    tools.compare(res1, -0.20955022, delta=1e-3)
    tools.compare(res2, -0.2250646, delta=1e-3)


def test_L1DecayRegularizer():
    """
    test L1DecayRegularizer
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            initializer = fluid.initializer.Constant(value=0.5)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc",
                                        input=data,
                                        size=10,
                                        param_attr=param_attrs)
            loss = fluid.layers.cross_entropy(input=y_predict, label=label)
            avg_loss = fluid.layers.mean(loss)
            optimizer = fluid.optimizer.Adagrad(
                learning_rate=0.1,
                regularization=fluid.regularizer.L1DecayRegularizer(
                    regularization_coeff=0.1))
            optimizer.minimize(avg_loss)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            x = np.ones(shape=(10, 1)).astype('float32')
            y = np.zeros(shape=(10, 1)).astype('int64')
            for i in range(10):
                res = exe.run(compiled_prog,
                              feed={"X": x,
                                    "label": y},
                              fetch_list=[avg_loss])[0][0]
            res1 = res

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            initializer = fluid.initializer.Constant(value=0.5)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc",
                                        input=data,
                                        size=10,
                                        param_attr=param_attrs)
            loss = fluid.layers.cross_entropy(input=y_predict, label=label)
            avg_loss = fluid.layers.mean(loss)
            optimizer = fluid.optimizer.Adagrad(learning_rate=0.1, )
            optimizer.minimize(avg_loss)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            x = np.ones(shape=(10, 1)).astype('float32')
            y = np.zeros(shape=(10, 1)).astype('int64')
            for i in range(10):
                res = exe.run(compiled_prog,
                              feed={"X": x,
                                    "label": y},
                              fetch_list=[avg_loss])[0][0]
            res2 = res
    print(res1)
    print(res2)
    tools.compare(res1, -0.20955022, delta=1e-3)
    tools.compare(res2, -0.2250646, delta=1e-3)


def test_L2Decay():
    """
    test L2Decay
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            initializer = fluid.initializer.Constant(value=0.5)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc",
                                        input=data,
                                        size=10,
                                        param_attr=param_attrs)
            loss = fluid.layers.cross_entropy(input=y_predict, label=label)
            avg_loss = fluid.layers.mean(loss)
            optimizer = fluid.optimizer.Adagrad(
                learning_rate=0.1,
                regularization=fluid.regularizer.L2Decay(
                    regularization_coeff=0.1))
            optimizer.minimize(avg_loss)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            x = np.ones(shape=(10, 1)).astype('float32')
            y = np.zeros(shape=(10, 1)).astype('int64')
            for i in range(10):
                res = exe.run(compiled_prog,
                              feed={"X": x,
                                    "label": y},
                              fetch_list=[avg_loss])[0][0]
            res1 = res

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            initializer = fluid.initializer.Constant(value=0.5)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc",
                                        input=data,
                                        size=10,
                                        param_attr=param_attrs)
            loss = fluid.layers.cross_entropy(input=y_predict, label=label)
            avg_loss = fluid.layers.mean(loss)
            optimizer = fluid.optimizer.Adagrad(learning_rate=0.1, )
            optimizer.minimize(avg_loss)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            x = np.ones(shape=(10, 1)).astype('float32')
            y = np.zeros(shape=(10, 1)).astype('int64')
            for i in range(10):
                res = exe.run(compiled_prog,
                              feed={"X": x,
                                    "label": y},
                              fetch_list=[avg_loss])[0][0]
            res2 = res
    print(res1)
    print(res2)
    tools.compare(res1, -0.21656528, delta=1e-3)
    tools.compare(res2, -0.2250646, delta=1e-3)


def test_L2DecayRegularizer():
    """
    test L2DecayRegularizer
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            initializer = fluid.initializer.Constant(value=0.5)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc",
                                        input=data,
                                        size=10,
                                        param_attr=param_attrs)
            loss = fluid.layers.cross_entropy(input=y_predict, label=label)
            avg_loss = fluid.layers.mean(loss)
            optimizer = fluid.optimizer.Adagrad(
                learning_rate=0.1,
                regularization=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.1))
            optimizer.minimize(avg_loss)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            x = np.ones(shape=(10, 1)).astype('float32')
            y = np.zeros(shape=(10, 1)).astype('int64')
            for i in range(10):
                res = exe.run(compiled_prog,
                              feed={"X": x,
                                    "label": y},
                              fetch_list=[avg_loss])[0][0]
            res1 = res

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            initializer = fluid.initializer.Constant(value=0.5)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc",
                                        input=data,
                                        size=10,
                                        param_attr=param_attrs)
            loss = fluid.layers.cross_entropy(input=y_predict, label=label)
            avg_loss = fluid.layers.mean(loss)
            optimizer = fluid.optimizer.Adagrad(learning_rate=0.1, )
            optimizer.minimize(avg_loss)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            x = np.ones(shape=(10, 1)).astype('float32')
            y = np.zeros(shape=(10, 1)).astype('int64')
            for i in range(10):
                res = exe.run(compiled_prog,
                              feed={"X": x,
                                    "label": y},
                              fetch_list=[avg_loss])[0][0]
            res2 = res
    print(res1)
    print(res2)
    tools.compare(res1, -0.21656528, delta=1e-3)
    tools.compare(res2, -0.2250646, delta=1e-3)
