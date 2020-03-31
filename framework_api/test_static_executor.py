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
"""test static executor."""
import paddle.fluid as fluid
import numpy as np
import tools
import paddle.fluid.compiler as compiler
import os
import time
import platform

def test_global_scope():
    """
    test global_scope
    :return:
    """
    fluid.global_scope().var("data").get_tensor().set(np.ones((1, 2)), fluid.CPUPlace())
    data = np.array(fluid.global_scope().find_var("data").get_tensor())
    tools.compare(data, [[1, 1]])


def test_scope_guard():
    """
    test scope_guard
    :return:
    """
    new_scope = fluid.Scope()
    with fluid.scope_guard(new_scope):
        fluid.global_scope().var("data").get_tensor().set(np.ones((1, 2)), fluid.CPUPlace())
        data = np.array(new_scope.find_var("data").get_tensor())
    tools.compare(data, [[1, 1]])


def test_Executor():
    """
    test Executor
    :return:
    """
    try:
        place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
        exe = fluid.Executor(place)
        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(train_program, startup_program):
                data = fluid.layers.data(name='X', shape=[1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = fluid.layers.mean(hidden)
                fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
                startup_program.random_seed = 1
                exe.run(startup_program)
                x = np.random.random(size=(10, 1)).astype('float32')
                for i in range(1000):
                    loss_data = exe.run(train_program,
                                         feed={"X": x},
                                         fetch_list=[loss.name])
        assert True
    except Exception:
        assert False


def test_Executor1():
    """
    test Executor with compileprogram
    :return:
    """
    try:
        place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
        exe = fluid.Executor(place)
        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(train_program, startup_program):
                data = fluid.layers.data(name='X', shape=[1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = fluid.layers.mean(hidden)
                fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
                startup_program.random_seed = 1
                exe.run(startup_program)
                x = np.random.random(size=(10, 1)).astype('float32')
                compiled_prog = compiler.CompiledProgram(
                    train_program).with_data_parallel(
                    loss_name=loss.name)
                if not fluid.is_compiled_with_cuda():
                    os.environ["CPU_NUM"] = "2"

                for i in range(1000):
                    loss_data = exe.run(compiled_prog,
                                         feed={"X": x},
                                         fetch_list=[loss.name])
        assert True
    except Exception:
        assert False


def test_Executor2():
    """
    test Executor with exe.close()
    :return:
    """
    try:
        place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
        exe = fluid.Executor(place)
        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(train_program, startup_program):
                data = fluid.layers.data(name='X', shape=[1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = fluid.layers.mean(hidden)
                fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
                startup_program.random_seed = 1
                exe.run(startup_program)
                x = np.random.random(size=(10, 1)).astype('float32')
                compiled_prog = compiler.CompiledProgram(
                    train_program).with_data_parallel(
                    loss_name=loss.name)
                if not fluid.is_compiled_with_cuda():
                    os.environ["CPU_NUM"] = "2"
                exe.close()
                for i in range(1000):
                    loss_data = exe.run(compiled_prog,
                                         feed={"X": x},
                                         fetch_list=[loss.name])
        assert False
    except Exception:
        assert True


def test_Executor3():
    """
    test Executor with run()
    :return:
    """
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
    exe = fluid.Executor(place)
    train_program = fluid.Program()
    startup_program = fluid.Program()
    startup_program.random_seed = 33
    train_program.random_seed = 33
    np.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
            startup_program.random_seed = 1
            exe.run(startup_program)
            x = np.ones(shape=(10, 1)).astype('float32')
            compiled_prog = compiler.CompiledProgram(
                train_program).with_data_parallel(
                loss_name=loss.name)
            if not fluid.is_compiled_with_cuda():
                os.environ["CPU_NUM"] = "2"
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
            for i in range(1000):
                loss_data = exe.run(compiled_prog,
                                     feed={"X": x},
                                     fetch_list=[loss.name])[0]
            if platform.system() == "Darwin" or platform.system() == "Linux":
                tools.compare(loss_data, [-1.9068239, -1.9068239])
            else:
                tools.compare(loss_data, [-1.9068239])


def test_Executor4():
    """
    test Executor with fetch_var_name feed_var_name
    :return:
    """
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
    exe = fluid.Executor(place)
    train_program = fluid.Program()
    startup_program = fluid.Program()
    startup_program.random_seed = 33
    train_program.random_seed = 33
    np.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
            startup_program.random_seed = 1
            exe.run(startup_program)
            x = np.ones(shape=(10, 1)).astype('float32')
            compiled_prog = compiler.CompiledProgram(
                train_program).with_data_parallel(
                loss_name=loss.name)
            if not fluid.is_compiled_with_cuda():
                os.environ["CPU_NUM"] = "2"
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
            for i in range(1000):
                loss_data = exe.run(compiled_prog,
                                     feed={"X": x},
                                     fetch_list=[loss.name],
                                     feed_var_name="f",
                                     fetch_var_name="c")[0]
            if platform.system() == "Darwin" or platform.system() == "Linux":
                tools.compare(loss_data, [-1.9068239, -1.9068239])
            else:
                tools.compare(loss_data, [-1.9068239])


def test_Executor5():
    """
    test Executor with use_program_cache=True
    :return:
    """
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
    exe = fluid.Executor(place)
    train_program = fluid.Program()
    startup_program = fluid.Program()
    startup_program.random_seed = 33
    train_program.random_seed = 33
    np.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
            startup_program.random_seed = 1
            exe.run(startup_program)
            x = np.ones(shape=(10, 1)).astype('float32')
            if not fluid.is_compiled_with_cuda():
                os.environ["CPU_NUM"] = "2"

            start = time.time()
            for i in range(1000):
                loss_data = exe.run(train_program,
                                    feed={"X": x},
                                    fetch_list=[loss.name],
                                    use_program_cache=True)[0]
            end1 = time.time() - start
            print(end1)
            tools.compare(loss_data, [-1.9068239])

    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
    exe = fluid.Executor(place)
    train_program = fluid.Program()
    startup_program = fluid.Program()
    startup_program.random_seed = 33
    train_program.random_seed = 33
    np.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
            startup_program.random_seed = 1
            exe.run(startup_program)
            x = np.ones(shape=(10, 1)).astype('float32')
            if not fluid.is_compiled_with_cuda():
                os.environ["CPU_NUM"] = "2"
            start = time.time()
            for i in range(1000):
                loss_data = exe.run(train_program,
                                    feed={"X": x},
                                    fetch_list=[loss.name],
                                    use_program_cache=False)[0]
            end2 = time.time() - start
            print(end2)
            tools.compare(loss_data, [-1.9068239])
    assert end2 > end1


def test_Executor6():
    """
    test Executor with return_numpy=False
    :return:
    """
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
    exe = fluid.Executor(place)
    train_program = fluid.Program()
    startup_program = fluid.Program()
    startup_program.random_seed = 33
    train_program.random_seed = 33
    np.random.seed(33)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
            startup_program.random_seed = 1
            exe.run(startup_program)
            x = np.ones(shape=(10, 1)).astype('float32')
            compiled_prog = compiler.CompiledProgram(
                train_program).with_data_parallel(
                loss_name=loss.name)
            if not fluid.is_compiled_with_cuda():
                os.environ["CPU_NUM"] = "2"
            for i in range(1000):
                loss_data = exe.run(compiled_prog,
                                     feed={"X": x},
                                     fetch_list=[loss.name],
                                     return_numpy=False)
            if "paddle.fluid.core_avx.LoDTensor" in loss_data.__str__():
                assert True
            else:
                assert False


def test_Executor7():
    """
    test Executor with scope=newscope
    :return:
    """
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
    exe = fluid.Executor(place)
    train_program = fluid.Program()
    startup_program = fluid.Program()
    startup_program.random_seed = 33
    train_program.random_seed = 33
    np.random.seed(33)
    fkscope = fluid.Scope()
    with fluid.scope_guard(fkscope):
        with fluid.unique_name.guard():
            with fluid.program_guard(train_program, startup_program):
                data = fluid.layers.data(name='X', shape=[1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = fluid.layers.mean(hidden)
                fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
                startup_program.random_seed = 1
                exe.run(startup_program)
                x = np.ones(shape=(10, 1)).astype('float32')
                compiled_prog = compiler.CompiledProgram(
                    train_program).with_data_parallel(
                    loss_name=loss.name)
                if not fluid.is_compiled_with_cuda():
                    os.environ["CPU_NUM"] = "2"
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
                for i in range(1000):
                    loss_data = exe.run(compiled_prog,
                                         feed={"X": x},
                                         fetch_list=[loss.name],
                                         scope=fkscope)[0]
                if platform.system() == "Darwin" or platform.system() == "Linux":
                    tools.compare(loss_data, [-1.9068239, -1.9068239])
                else:
                    tools.compare(loss_data, [-1.9068239])

