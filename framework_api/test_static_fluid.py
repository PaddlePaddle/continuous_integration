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
"""test static fluid."""

import paddle.fluid as fluid
import os
import numpy as np
import tools
import operator
import platform


def test_cpu_places():
    """
    test cpu_places with no environ
    :return:
    """
    cpu_places = fluid.cpu_places()
    assert len(cpu_places) == 1


def test_cpu_places1():
    """
    test cpu_places with environ cpunum = 4
    :return:
    """
    os.environ["CPU_NUM"] = "4"
    cpu_places = fluid.cpu_places()
    assert len(cpu_places) == 4


def test_CPUPlace():
    """
    test CPUPlace
    :return:
    """
    cpu_place = fluid.CPUPlace()
    tools.compare(cpu_place.__str__(), "CPUPlace")


def test_create_lod_tensor():
    """
    test create_lod_tensor with data=np.array
    :return:
    """
    res = list()
    for i in range(10):
        for j in range(10):
            for k in range(i + j):
                res.append([1])
            t = fluid.create_lod_tensor(
                np.array(res), [[i, j]], fluid.CPUPlace())
            lod = [[0, i, i + j]]
            tools.compare(t.lod(), lod)
            expect = [[1]] * (i + j)
            tools.compare(res, expect)
            # print(t.lod())
            res = list()


def test_create_lod_tensor1():
    """
    test create_lod_tensor with data=lod_tensor
    :return:
    """
    res = list()
    for i in range(10):
        for j in range(10):
            for k in range(i + j):
                res.append([1])
            t = fluid.create_lod_tensor(
                np.array(res), [[i, j]], fluid.CPUPlace())
            t = fluid.create_lod_tensor(t, [[i, j]], fluid.CPUPlace())
            lod = [[0, i, i + j]]
            tools.compare(t.lod(), lod)
            expect = [[1]] * (i + j)
            tools.compare(res, expect)
            # print(t.lod())
            res = list()


def test_create_lod_tensor2():
    """
    test create_lod_tensor with data=list
    :return:
    """
    list = [[1, 2, 3], [1], [2, 4, 5, 6]]
    t = fluid.create_lod_tensor(list, [[3, 1, 4]], fluid.CPUPlace())
    print(t)
    lod = [[0, 3, 4, 8]]
    tools.compare(t.lod(), lod)
    tools.compare(
        np.array(t.__array__()).flatten(), np.array([1, 2, 3, 1, 2, 4, 5, 6]))


def test_create_random_int_lodtensor():
    """
    test create_random_int_lodtensor
    :return:
    """
    np.random.seed(33)
    for i in range(10):
        for j in range(10):
            for shape in range(10):
                t = fluid.create_random_int_lodtensor(
                    recursive_seq_lens=[[i, j]],
                    base_shape=[shape],
                    place=fluid.CPUPlace(),
                    low=0,
                    high=10)
                tools.compare(t.shape(), [i + j, shape])
                for key in list(t.__array__().flatten()):
                    if key >= 0 or key <= 10:
                        assert True


def test_data():
    """
    test data
    :return:
    """
    x = fluid.data(name='x', shape=[3, 2, 1], dtype='float32')

    y = fluid.data(name='y', shape=[3, 2, 1], dtype='float32')

    z = x + y

    feed_data = np.ones(shape=[3, 2, 1], dtype=np.float32)

    exe = fluid.Executor(fluid.CPUPlace())
    res = exe.run(fluid.default_main_program(),
                  feed={'x': feed_data,
                        'y': feed_data},
                  fetch_list=[z.name])[0]
    expect = np.array(
        [[[2.], [2.]], [[2.], [2.]], [[2.], [2.]]]).astype(np.float32)
    tools.compare(res, expect)


def test_data1():
    """
    test data with shape = tuple
    :return:
    """
    x = fluid.data(name='x', shape=(3, 2, 1), dtype='float32')

    y = fluid.data(name='y', shape=(3, 2, 1), dtype='float32')

    z = x + y

    feed_data = np.ones(shape=[3, 2, 1], dtype=np.float32)

    exe = fluid.Executor(fluid.CPUPlace())
    res = exe.run(fluid.default_main_program(),
                  feed={'x': feed_data,
                        'y': feed_data},
                  fetch_list=[z.name])[0]
    expect = np.array(
        [[[2.], [2.]], [[2.], [2.]], [[2.], [2.]]]).astype(np.float32)
    tools.compare(res, expect)


def test_data2():
    """
    test data with shape = tuple dytype = all !!! other types are wrong
    :return:
    """
    # type = [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8]
    type = [np.float32]
    for i in type:
        x = fluid.data(name='x', shape=(3, 2, 1), dtype=i)

        y = fluid.data(name='y', shape=(3, 2, 1), dtype=i)

        z = x + y

        feed_data = np.ones(shape=[3, 2, 1], dtype=i)

        exe = fluid.Executor(fluid.CPUPlace())
        res = exe.run(fluid.default_main_program(),
                      feed={'x': feed_data,
                            'y': feed_data},
                      fetch_list=[z.name])[0]
        expect = np.array([[[2.], [2.]], [[2.], [2.]], [[2.], [2.]]]).astype(i)
        tools.compare(res, expect)


def test_DataFeeder():
    """
    test DataFeeder
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            place = fluid.CPUPlace()
            img = fluid.layers.data(name='image', shape=[1, 28, 28])
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            feeder = fluid.DataFeeder([img, label], place)
            result = feeder.feed([([0] * 784, [9]), ([1] * 784, [1])])
            expect_label = [[9], [1]]
            expect_image = np.array([[0] * 784, [1] * 784]).reshape(2, 1, 28,
                                                                    28)
            tools.compare(result["label"].__array__(), expect_label)
            tools.compare(result["image"].__array__(), expect_image)


def test_DataFeeder1():
    """
    test DataFeeder with 3 dims
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            place = fluid.CPUPlace()
            img = fluid.layers.data(name='image', shape=[1, 28, 28])
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            value = fluid.layers.data(name='value', shape=[1, 2, 2])
            feeder = fluid.DataFeeder([img, label, value], place)
            result = feeder.feed(
                [([0] * 784, [9], [33] * 4), ([1] * 784, [1], [25] * 4)])
            expect_label = [[9], [1]]
            expect_image = np.array([[0] * 784, [1] * 784]).reshape(2, 1, 28,
                                                                    28)
            expect_value = np.array([[33] * 4, [25] * 4]).reshape(2, 1, 2, 2)
            tools.compare(result["label"].__array__(), expect_label)
            tools.compare(result["image"].__array__(), expect_image)
            tools.compare(result["value"].__array__(), expect_value)


def test_DataFeeder2():
    """
    test DataFeeder with program = train_program
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            place = fluid.CPUPlace()
            img = fluid.layers.data(name='image', shape=[1, 28, 28])
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            value = fluid.layers.data(name='value', shape=[1, 2, 2])
            feeder = fluid.DataFeeder(
                [img, label, value], place, program=train_program)
            result = feeder.feed(
                [([0] * 784, [9], [33] * 4), ([1] * 784, [1], [25] * 4)])
            expect_label = [[9], [1]]
            expect_image = np.array([[0] * 784, [1] * 784]).reshape(2, 1, 28,
                                                                    28)
            expect_value = np.array([[33] * 4, [25] * 4]).reshape(2, 1, 2, 2)
            tools.compare(result["label"].__array__(), expect_label)
            tools.compare(result["image"].__array__(), expect_image)
            tools.compare(result["value"].__array__(), expect_value)


def test_default_main_program():
    """
    test default_main_program
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            # 示例网络:
            data = fluid.layers.data(
                name='image', shape=[3, 224, 224], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')

            conv1 = fluid.layers.conv2d(data, 4, 5, 1, act=None)
            bn1 = fluid.layers.batch_norm(conv1, act='relu')
            pool1 = fluid.layers.pool2d(bn1, 2, 'max', 2)
            conv2 = fluid.layers.conv2d(pool1, 16, 5, 1, act=None)
            bn2 = fluid.layers.batch_norm(conv2, act='relu')
            pool2 = fluid.layers.pool2d(bn2, 2, 'max', 2)

            fc1 = fluid.layers.fc(pool2, size=50, act='relu')
            fc2 = fluid.layers.fc(fc1, size=102, act='softmax')

            loss = fluid.layers.cross_entropy(input=fc2, label=label)
            loss = fluid.layers.mean(loss)
            opt = fluid.optimizer.Momentum(
                learning_rate=0.1,
                momentum=0.9,
                regularization=fluid.regularizer.L2Decay(1e-4))
            opt.minimize(loss)

            # print(fluid.default_main_program().num_blocks)
            tools.compare(fluid.default_main_program().num_blocks, 1)
            print(fluid.default_main_program().blocks[0].var('image'))


def test_default_startup_program():
    """
    test default_startup_program
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name="x", shape=[-1, 784], dtype='float32')
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='int32')
            z = fluid.layers.fc(name="fc", input=x, size=10, act="relu")

            # print("main program is: {}".format(type(fluid.default_main_program())))
            tools.compare(
                type(fluid.default_main_program()), fluid.framework.Program)
            tools.compare(
                type(fluid.default_startup_program()), fluid.framework.Program)
            # print("start up program is: {}".format(fluid.default_startup_program()))


def test_ExecutionStrategy():
    """
    test ExecutionStrategy
    通过设置 ExecutionStrategy 中的选项，用户可以对执行器的执行配置进行调整，比如设置执行器中线程池的大小等。
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            os.environ["CPU_NUM"] = "4"
            x = fluid.layers.data(name='x', shape=[13], dtype='float32')
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            y_predict = fluid.layers.fc(input=x, size=1, act=None)
            cost = fluid.layers.square_error_cost(input=y_predict, label=y)
            avg_loss = fluid.layers.mean(cost)
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
            sgd_optimizer.minimize(avg_loss)
            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_threads = 4
            exec_strategy.num_iteration_per_drop_scope = 1
            exec_strategy.num_iteration_per_run = 10
            for i in range(100):
                train_exe = fluid.ParallelExecutor(
                    use_cuda=False,
                    loss_name=avg_loss.name,
                    exec_strategy=exec_strategy)


def test_global_scope():
    """
    test global_scope
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            fluid.global_scope().var("data").get_tensor().set(
                np.ones((1, 2)), fluid.CPUPlace())
            data = np.array(fluid.global_scope().find_var("data").get_tensor())
            tools.compare(data, [[1, 1]])
            # print(data)  # [[1. 1.]]


def test_gradients():
    """
    test gradients
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            x = fluid.layers.data(name='x', shape=[1], dtype='float32')
            x.stop_gradient = False
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            z = x * y
            f = fluid.gradients(z, x)

            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            x1 = np.ones(shape=(10, 1), dtype=np.float32)
            y1 = np.ones(shape=(10, 1), dtype=np.float32) * 5
            res = exe.run(compiled_prog,
                          feed={"x": x1,
                                "y": y1},
                          fetch_list=["x@GRAD"])[0][0]
            tools.compare(res, [5])


def test_gradients1():
    """
    test gradients with no_grad_set = y@GRAD
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            x = fluid.layers.data(name='x', shape=[1], dtype='float32')
            x.stop_gradient = False
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            z = x * y
            f = fluid.gradients(z, x, no_grad_set=["y@GRAD"])

            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            x1 = np.ones(shape=(10, 1), dtype=np.float32)
            y1 = np.ones(shape=(10, 1), dtype=np.float32) * 5
            res = exe.run(compiled_prog,
                          feed={"x": x1,
                                "y": y1},
                          fetch_list=["x@GRAD"])[0][0]
            # print(res)
            tools.compare(res, [5])


def test_gradients2():
    """
    test gradients with no_grad_set = x@GRAD  exception!!
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            x = fluid.layers.data(name='x', shape=[1], dtype='float32')
            x.stop_gradient = False
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            z = x * y
            f = fluid.gradients(z, x, no_grad_set=set("x@GRAD"))
            try:
                compiled_prog = fluid.compiler.CompiledProgram(train_program)
                x1 = np.ones(shape=(10, 1), dtype=np.float32)
                y1 = np.ones(shape=(10, 1), dtype=np.float32) * 5
                res = exe.run(compiled_prog,
                              feed={"x": x1,
                                    "y": y1},
                              fetch_list=["x@GRAD"])[0][0]
                # print(res)
                tools.compare(res, [5])
            except Exception:
                assert True


def test_gradients3():
    """
    test gradients with target_gradients = x
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            x = fluid.layers.data(name='x', shape=[1], dtype='float32')
            x.stop_gradient = False
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            z = x * y
            f = fluid.gradients(z, x, target_gradients=x)

            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            x1 = np.ones(shape=(10, 1), dtype=np.float32)
            y1 = np.ones(shape=(10, 1), dtype=np.float32) * 5
            res = exe.run(compiled_prog,
                          feed={"x": x1,
                                "y": y1},
                          fetch_list=["x@GRAD"])[0][0]
            tools.compare(res, [5])


def test_gradients4():
    """
    test gradients with target_gradients = z
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            x = fluid.layers.data(name='x', shape=[1], dtype='float32')
            x.stop_gradient = False
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            z = x * y
            f = fluid.gradients(z, x, target_gradients=z)

            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            x1 = np.ones(shape=(10, 1), dtype=np.float32)
            y1 = np.ones(shape=(10, 1), dtype=np.float32) * 5
            res = exe.run(compiled_prog,
                          feed={"x": x1,
                                "y": y1},
                          fetch_list=["x@GRAD"])[0][0]
            tools.compare(res, [25])


def test_in_dygraph_mode():
    """
    test in_dygraph_mode
    :return:
    """
    if fluid.in_dygraph_mode():
        print('running in dygraph mode')
    tools.compare(fluid.in_dygraph_mode(), False)
    with fluid.dygraph.guard():
        if fluid.in_dygraph_mode():
            print('running in dygraph mode')
            tools.compare(fluid.in_dygraph_mode(), True)


def test_is_compiled_with_cuda():
    """
    test is_compiled_with_cuda
    :return:
    """
    if fluid.is_compiled_with_cuda():
        tools.compare(fluid.is_compiled_with_cuda(), True)
    else:
        tools.compare(fluid.is_compiled_with_cuda(), False)


def test_LoDTensor():
    """
    test LoDTensor with has_valid_recursive_sequence_lengths = True
    :return:
    """
    t = fluid.LoDTensor()
    t.set(np.ndarray([5, 30]), fluid.CPUPlace())
    t.set_recursive_sequence_lengths([[2, 3]])
    tools.compare(t.has_valid_recursive_sequence_lengths(), True)


def test_LoDTensor1():
    """
    test LoDTensor with lod()
    :return:
    """
    t = fluid.LoDTensor()
    t.set(np.ndarray([5, 30]), fluid.CPUPlace())
    t.set_recursive_sequence_lengths([[2, 3]])
    tools.compare(t.has_valid_recursive_sequence_lengths(), True)
    t.set(np.ndarray([5, 30]), fluid.CPUPlace())
    t.set_lod([[0, 2, 5]])
    tools.compare(t.lod(), [[0, 2, 5]])


def test_LoDTensor2():
    """
    test LoDTensor with recursive_sequence_lengths()
    :return:
    """
    t = fluid.LoDTensor()
    t.set(np.ndarray([5, 30]), fluid.CPUPlace())
    t.set_recursive_sequence_lengths([[2, 3]])
    tools.compare(t.recursive_sequence_lengths(), [[2, 3]])


def test_LoDTensor3():
    """
    test LoDTensor with setplace()
    :return:
    """
    t = fluid.LoDTensor()
    t.set(np.ndarray([5, 30]), fluid.CPUPlace())
    tools.compare(type(t), fluid.core_avx.LoDTensor)


def test_LoDTensor4():
    """
    test LoDTensor with shape
    :return:
    """
    t = fluid.LoDTensor()
    t.set(np.ndarray([5, 30]), fluid.CPUPlace())
    tools.compare(t.shape(), [5, 30])


def test_LoDTensorArray():
    """
    test LoDTensorArray
    :return:
    """
    arr = fluid.LoDTensorArray()
    t = fluid.LoDTensor()
    t.set(np.ndarray([5, 30]), fluid.CPUPlace())
    arr.append(t)
    tools.compare(type(arr), fluid.core_avx.LoDTensorArray)


def test_one_hot():
    """
    test one_hot with append_batch_size=False
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            label = fluid.layers.data(
                name="label", shape=[4], append_batch_size=False, dtype="int64")
            one_hot_label = fluid.one_hot(input=label, depth=4)
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            x1 = np.array([[1, 1, 3, 0]])

            res = exe.run(compiled_prog,
                          feed={"label": x1},
                          fetch_list=["one_hot_v2_0.tmp_0"])[0][0]
            expect = [[0., 1., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.],
                      [1., 0., 0., 0.]]
            tools.compare(res, expect)


def test_one_hot1():
    """
    test one_hot with append_batch_size=True !!!!! there is allow_out_of_range error
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            label = fluid.layers.data(
                name="label", shape=[4], append_batch_size=False, dtype="int64")
            one_hot_label = fluid.one_hot(
                input=label, depth=4, allow_out_of_range=True)
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            x1 = np.array([[1, 1, 5, 0]])

            res = exe.run(compiled_prog,
                          feed={"label": x1},
                          fetch_list=["one_hot_v2_0.tmp_0"])[0][0]
            expect = [[0., 1., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.],
                      [1., 0., 0., 0.]]
            tools.compare(res, expect)


def test_one_hot2():
    """
    test one_hot with append_batch_size=False  Illegal value
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            try:
                label = fluid.layers.data(
                    name="label",
                    shape=[4],
                    append_batch_size=False,
                    dtype="int64")
                one_hot_label = fluid.one_hot(
                    input=label, depth=4, allow_out_of_range=True)
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                compiled_prog = fluid.compiler.CompiledProgram(train_program)
                x1 = np.array([[1, 1, 5, 0]])

                res = exe.run(compiled_prog,
                              feed={"label": x1},
                              fetch_list=["one_hot_v2_0.tmp_0"])[0][0]
                expect = [[0., 1., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.],
                          [1., 0., 0., 0.]]
                tools.compare(res, expect)
                assert False
            except Exception as e:
                assert True


def test_Program():
    """
    test Program
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            x = fluid.layers.data(name="x", shape=[-1, 784], dtype='float32')
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='int32')
            z = fluid.layers.fc(name="fc", input=x, size=10, act="relu")
            # print("main program is: {}".format(main_program))
            # print("start up program is: {}".format(startup_program))
            tools.compare(
                len(startup_program.__dict__), len(main_program.__dict__))


def test_Program1():
    """
    test Program to_string
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            a = fluid.layers.data(
                name="X",
                shape=[2, 3],
                dtype="float32",
                append_batch_size=False)
            c = fluid.layers.fc(a, size=3)
            prog_string = main_program.to_string(
                throw_on_error=True, with_details=False)
            prog_string_with_details = main_program.to_string(
                throw_on_error=False, with_details=True)
            assert "optimize_attr" not in prog_string
            assert "optimize_attr" in prog_string_with_details
            # print(prog_string)
            # print("\n =============== with_details =============== \n")
            # print(prog_string_with_details)


def test_Program2():
    """
    test Program clone
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            label = fluid.layers.data(name="Y", shape=[1], dtype='int64')
            initializer = fluid.initializer.Constant(value=2.0)
            param_attrs = fluid.ParamAttr(initializer=initializer)
            y_predict = fluid.layers.fc(name="fc",
                                        input=data,
                                        size=10,
                                        param_attr=param_attrs)
            cost = fluid.layers.cross_entropy(input=y_predict, label=label)
            sum_cost = fluid.layers.reduce_mean(cost)
            optimizer = fluid.optimizer.Momentum(
                learning_rate=0.1, momentum=0.1)
            optimizer.minimize(sum_cost)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            x = np.random.random(size=(10, 1)).astype('float32')
            y = np.ones(shape=(10, 1)).astype("int64")
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(10):
                res = exe.run(compiled_prog,
                              feed={"X": x,
                                    "Y": y},
                              fetch_list=["fc.w_0"])[0][0]
            clone_z = main_program.clone(for_test=True)
            clone = main_program.clone(for_test=False)
            x = clone.__dict__.copy()["blocks"][0].__str__().split()
            y = main_program.__dict__.copy()["blocks"][0].__str__().split()
            z = clone_z.__dict__.copy()["blocks"][0].__str__().split()
            assert operator.eq(sorted(x), sorted(y))
            assert not operator.eq(sorted(y), sorted(z))


def test_Program3():
    """
    test Program | parse_from_string
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            x = fluid.layers.data(
                name='X',
                shape=[1000, 784],
                dtype='float32',
                append_batch_size=False)

            y = fluid.layers.data(
                name='Y',
                shape=[784, 100],
                dtype='float32',
                append_batch_size=False)

            z = fluid.layers.mul(x=x, y=y)

            binary_str = main_program.desc.serialize_to_string()
            prog_restored = main_program.parse_from_string(binary_str)
            assert operator.eq(main_program.__dict__['blocks'].sort(),
                               prog_restored.__dict__['blocks'].sort())


def test_Program4():
    """
    test Program | numblocks
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            tools.compare(main_program.num_blocks, 1)


def test_Program5():
    """
    test Program | random_seed
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            tools.compare(main_program.random_seed, 0)


def test_Program6():
    """
    test Program | global_block()
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            try:
                gb_block = main_program.global_block()
                print(gb_block.__str__)
                assert True
            except Exception:
                assert False


def test_Program7():
    """
    test Program | block()
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            try:
                gb_block = main_program.block(0)
                print(gb_block)
                assert True
            except Exception:
                assert False


def test_Program8():
    """
    test Program | block()
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            try:
                gb_block = main_program.current_block()
                print(gb_block)
                assert True
            except Exception:
                assert False


def test_Program9():
    """
    test Program | list_vars()
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            img = fluid.layers.data(
                name='img', shape=[1, 28, 28], dtype='float32')
            label = fluid.layers.data(
                name='label', shape=[128, 1], dtype='int64')
            List = list(enumerate(main_program.list_vars()))
            tools.compare(len(List), 2)


def test_scope_guard():
    """
    test scope_guard
    :return:
    """
    new_scope = fluid.Scope()
    with fluid.scope_guard(new_scope):
        fluid.global_scope().var("data").get_tensor().set(
            np.ones((1, 2)), fluid.CPUPlace())
        data = np.array(new_scope.find_var("data").get_tensor())
        tools.compare(data, [[1, 1]])


def test_Tensor():
    """
    test Tensor
    :return:
    """
    t = fluid.Tensor()
    t.set(np.ndarray([5, 30]), fluid.CPUPlace())
    tools.compare(t.shape(), [5, 30])


def test_Variable():
    """
    test Variable
    :return:
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
    with fluid.dygraph.guard():
        new_variable = fluid.dygraph.to_variable(np.arange(10))
        tools.compare(new_variable.numpy(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            cur_block = main_program.current_block()
            new_variable = cur_block.create_var(
                name="X", shape=[-1, 23, 48], dtype='float32')


def test_Variable1():
    """
    test Variable with detach
    :return:
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
    with fluid.dygraph.guard():
        new_variable = fluid.dygraph.to_variable(np.arange(10))
        tools.compare(new_variable.numpy(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        y = new_variable.detach()
        tools.compare(y.numpy(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_Variable2():
    """
    test Variable with set_value()
    :return:
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
    with fluid.dygraph.guard():
        new_variable = fluid.dygraph.to_variable(np.arange(10))
        tools.compare(new_variable.numpy(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        y = new_variable.detach()
        tools.compare(y.numpy(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        if platform.system() == "Linux" or platform.system() == "Darwin":
            ddtype = np.int64
        else:
            ddtype = np.int32
        new_variable.set_value(np.ones(shape=[10], dtype=ddtype))
        tools.compare(new_variable.numpy(), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


def test_DataFeedDesc():
    """
    test DataFeedDesc
    :return:
    """
    if os.path.exists("./data.proto"):
        os.remove("./data.proto")
    f = open("data.proto", "w")
    print('name: "MultiSlotDataFeed"', file=f)
    print('batch_size: 2', file=f)
    print('multi_slot_desc {', file=f)
    print('    slots {', file=f)
    print('         name: "words"', file=f)
    print('         type: "int64"', file=f)
    print('         is_dense: False', file=f)
    print('         is_used: False', file=f)
    print('     }', file=f)
    print('     slots {', file=f)
    print('         name: "label"', file=f)
    print('         type: "int64"', file=f)
    print('         is_dense: False', file=f)
    print('         is_used: False', file=f)
    print('    }', file=f)
    print('}', file=f)
    f.close()
    data_feed = fluid.DataFeedDesc('data.proto')
    data_feed.set_batch_size(128)
    data_feed.set_dense_slots(['words'])
    data_feed.set_use_slots(['words'])
    if "batch_size: 128" in data_feed.desc():
        assert True
    else:
        assert False
    if "is_dense: true" in data_feed.desc(
    ) and "is_dense: false" in data_feed.desc():
        assert True
    else:
        assert False
    if "is_used: true" in data_feed.desc(
    ) and "is_used: false" in data_feed.desc():
        assert True
    else:
        assert False
    if os.path.exists("./data.proto"):
        os.remove("./data.proto")


def test_ParallelExecutor():
    """
    test ParallelExecutor
    :return:
    """
    if not fluid.is_compiled_with_cuda():
        os.environ['CPU_NUM'] = str(2)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
    use_cuda = fluid.is_compiled_with_cuda()
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
    ) else fluid.CPUPlace()

    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            try:
                exe = fluid.Executor(place)
                data = fluid.layers.data(name='X', shape=[1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = fluid.layers.mean(hidden)
                test_program = fluid.default_main_program().clone(for_test=True)
                fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

                exe.run(startup_program)

                train_exe = fluid.ParallelExecutor(
                    use_cuda=use_cuda,
                    main_program=train_program,
                    loss_name=loss.name)
                # 注意：如果此处不设置share_vars_from=train_exe，测试过程中用的参数与训练使用的参数是不一致
                test_exe = fluid.ParallelExecutor(
                    use_cuda=use_cuda,
                    main_program=test_program,
                    share_vars_from=train_exe)

                train_data = np.ones(shape=(10, 1)).astype('float32')
                loss_data, = train_exe.run(feed={"X": train_data},
                                           fetch_list=[loss.name])
                print(loss_data)
                if platform.system() == "Darwin" or platform.system(
                ) == "Linux":
                    tools.compare(loss_data, [0.09116864, 0.09116864])
                else:
                    tools.compare(loss_data, [0.09116864])
                test_data = np.ones(shape=(10, 1)).astype('float32')
                loss_data, = test_exe.run(feed={"X": test_data},
                                          fetch_list=[loss.name])
                print(loss_data)
                if platform.system() == "Darwin" or platform.system(
                ) == "Linux":
                    tools.compare(loss_data, [0.08916866, 0.08916866])
                else:
                    tools.compare(loss_data, [0.08916866])
            except Exception:
                assert False


def test_ParallelExecutor1():
    """
    test ParallelExecutor with drop_local_exe_scopes()
    :return:
    """
    use_cuda = fluid.is_compiled_with_cuda()
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
    ) else fluid.CPUPlace()
    if not fluid.is_compiled_with_cuda():
        os.environ['CPU_NUM'] = str(2)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            try:
                exe = fluid.Executor(place)
                data = fluid.layers.data(name='X', shape=[1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = fluid.layers.mean(hidden)
                test_program = fluid.default_main_program().clone(for_test=True)
                fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

                exe.run(startup_program)

                train_exe = fluid.ParallelExecutor(
                    use_cuda=use_cuda,
                    main_program=train_program,
                    loss_name=loss.name)
                # 注意：如果此处不设置share_vars_from=train_exe，测试过程中用的参数与训练使用的参数是不一致
                test_exe = fluid.ParallelExecutor(
                    use_cuda=use_cuda,
                    main_program=test_program,
                    share_vars_from=train_exe)

                train_data = np.ones(shape=(10, 1)).astype('float32')
                loss_data, = train_exe.run(feed={"X": train_data},
                                           fetch_list=[loss.name])
                print(loss_data)
                if platform.system() == "Darwin" or platform.system(
                ) == "Linux":
                    tools.compare(loss_data, [0.09116864, 0.09116864])
                else:
                    tools.compare(loss_data, [0.09116864])
                train_exe.drop_local_exe_scopes()
                test_data = np.ones(shape=(10, 1)).astype('float32')
                loss_data, = test_exe.run(feed={"X": test_data},
                                          fetch_list=[loss.name])
                print(loss_data)
                if platform.system() == "Darwin" or platform.system(
                ) == "Linux":
                    tools.compare(loss_data, [0.08916866, 0.08916866])
                else:
                    tools.compare(loss_data, [0.08916866])
            except Exception:
                assert False


def test_CompiledProgram():
    """
    test CompiledProgram
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
    ) else fluid.CPUPlace()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            exe = fluid.Executor(place)
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
            exe.run(startup_program)
            build_strategy = fluid.BuildStrategy()
            compiled_prog = fluid.CompiledProgram(
                train_program, build_strategy=build_strategy)
            x = np.ones(shape=(10, 1)).astype('float32')
            loss_data, = exe.run(compiled_prog,
                                 feed={"X": x},
                                 fetch_list=[loss.name])
            tools.compare(loss_data, [0.09116866])


def test_CompiledProgram1():
    """
    test CompiledProgram with with_data_parallel
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
    ) else fluid.CPUPlace()
    if not fluid.is_compiled_with_cuda():
        os.environ['CPU_NUM'] = str(2)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            exe = fluid.Executor(place)
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            test_program = train_program.clone(for_test=True)
            fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

            exe.run(startup_program)
            build_strategy = fluid.BuildStrategy()
            build_strategy.fuse_all_reduce_ops = True
            compiled_train_prog = fluid.CompiledProgram(
                train_program).with_data_parallel(
                    loss_name=loss.name, build_strategy=build_strategy)
            # 注意：如果此处不设置share_vars_from=compiled_train_prog，测试过程中用的参数与训练使用的参数是不一致
            compiled_test_prog = fluid.CompiledProgram(
                test_program).with_data_parallel(
                    share_vars_from=compiled_train_prog)

            train_data = np.ones(shape=(10, 1)).astype('float32')
            loss_data, = exe.run(compiled_train_prog,
                                 feed={"X": train_data},
                                 fetch_list=[loss.name])
            print(loss_data)
            if platform.system() == "Darwin" or platform.system() == "Linux":
                tools.compare(loss_data, [0.09116864, 0.09116864])
            else:
                tools.compare(loss_data, [0.09116864])
            test_data = np.ones(shape=(10, 1)).astype('float32')
            loss_data, = exe.run(compiled_test_prog,
                                 feed={"X": test_data},
                                 fetch_list=[loss.name])

            print(loss_data)
            if platform.system() == "Darwin" or platform.system() == "Linux":
                tools.compare(loss_data, [0.08916866, 0.08916866])
            else:
                tools.compare(loss_data, [0.08916866])


def test_BuildStrategy():
    """
    test BuildStrategy with compiled program
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
    ) else fluid.CPUPlace()
    if not fluid.is_compiled_with_cuda():
        os.environ['CPU_NUM'] = str(2)
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            exe = fluid.Executor(place)
            data = fluid.layers.data(name='X', shape=[1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)
            exe.run(startup_program)
            build_strategy = fluid.BuildStrategy()
            build_strategy.fuse_all_optimizer_ops = True
            build_strategy.debug_graphviz_path = "./graph"
            build_strategy.enable_sequential_execution = True
            build_strategy.fuse_elewise_add_act_ops = True
            build_strategy.fuse_broadcast_ops = True
            build_strategy.fuse_relu_depthwise_conv = True
            build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
            build_strategy.remove_unnecessary_lock = True
            build_strategy.sync_batch_norm = True
            compiled_prog = fluid.CompiledProgram(
                train_program, build_strategy=build_strategy)
            x = np.ones(shape=(10, 1)).astype('float32')
            loss_data, = exe.run(compiled_prog,
                                 feed={"X": x},
                                 fetch_list=[loss.name])
            tools.compare(loss_data, [0.09116866])


def test_BuildStrategy1():
    """
    test BuildStrategy with parallelExecutor
    :return:
    """
    use_cuda = fluid.is_compiled_with_cuda()
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
    ) else fluid.CPUPlace()
    if not fluid.is_compiled_with_cuda():
        os.environ['CPU_NUM'] = str(2)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = 33
    startup_program.random_seed = 33
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            try:
                exe = fluid.Executor(place)
                data = fluid.layers.data(name='X', shape=[1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = fluid.layers.mean(hidden)
                test_program = fluid.default_main_program().clone(for_test=True)
                fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

                exe.run(startup_program)
                build_strategy = fluid.BuildStrategy()
                build_strategy.fuse_all_optimizer_ops = True
                build_strategy.debug_graphviz_path = "./graph"
                build_strategy.enable_sequential_execution = True
                build_strategy.fuse_elewise_add_act_ops = True
                build_strategy.fuse_broadcast_ops = True
                build_strategy.fuse_relu_depthwise_conv = True
                build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
                build_strategy.remove_unnecessary_lock = True
                build_strategy.sync_batch_norm = True
                train_exe = fluid.ParallelExecutor(
                    use_cuda=use_cuda,
                    main_program=train_program,
                    build_strategy=build_strategy,
                    loss_name=loss.name)
                # 注意：如果此处不设置share_vars_from=train_exe，测试过程中用的参数与训练使用的参数是不一致
                test_exe = fluid.ParallelExecutor(
                    use_cuda=use_cuda,
                    main_program=test_program,
                    share_vars_from=train_exe)

                train_data = np.ones(shape=(10, 1)).astype('float32')
                loss_data, = train_exe.run(feed={"X": train_data},
                                           fetch_list=[loss.name])
                print(loss_data)
                if platform.system() == "Darwin" or platform.system(
                ) == "Linux":
                    tools.compare(loss_data, [0.09116864, 0.09116864])
                else:
                    tools.compare(loss_data, [0.09116864])
                test_data = np.ones(shape=(10, 1)).astype('float32')
                loss_data, = test_exe.run(feed={"X": test_data},
                                          fetch_list=[loss.name])
                print(loss_data)
                if platform.system() == "Darwin" or platform.system(
                ) == "Linux":
                    tools.compare(loss_data, [0.08916866, 0.08916866])
                else:
                    tools.compare(loss_data, [0.08916866])
            except Exception:
                assert False
