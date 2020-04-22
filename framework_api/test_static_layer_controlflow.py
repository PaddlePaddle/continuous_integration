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
"""test static layer controlflow."""
import paddle.fluid as fluid
import numpy as np
import tools


def test_array_length():
    """
    test array length
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            # define value
            value = 13
            tmp = fluid.layers.zeros(shape=[10], dtype='int32')
            i = fluid.layers.fill_constant(
                shape=[1], dtype='int64', value=value)
            arr = fluid.layers.array_write(tmp, i=i)
            arr_len = fluid.layers.array_length(arr)
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            np.random.seed(11)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[arr_len])[0][0]
            tools.compare(res, value + 1)


def test_array_read():
    """
    test array read
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            arr = fluid.layers.create_array(dtype='float32')
            tmp = fluid.layers.fill_constant(
                shape=[3, 2], dtype='int64', value=5)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
            # tmp是shape为[3,2]的Tensor，将其写入空数组arr的下标10的位置，则arr的长度变为11
            arr = fluid.layers.array_write(tmp, i, array=arr)
            # 读取arr的下标10的位置的数据
            item = fluid.layers.array_read(arr, i)
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[item])[0]
            expect = np.array([[5, 5], [5, 5], [5, 5]])
            tools.compare(res, expect)


def test_array_write():
    """
    test array write
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            tmp = fluid.layers.fill_constant(
                shape=[3, 2], dtype='int64', value=5)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
            # 将tmp写入数组arr下标为10的位置，并返回arr
            arr = fluid.layers.array_write(tmp, i=i)
            tmp = fluid.layers.fill_constant(
                shape=[3, 2], dtype='int64', value=7)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
            # 将tmp写入数组arr下标为10的位置，并返回arr
            arr = fluid.layers.array_write(tmp, i=i)
            # 读取arr的下标10的位置的数据
            item = fluid.layers.array_read(arr, i)
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[item])[0]
            expect = np.array([[7, 7], [7, 7], [7, 7]])
            tools.compare(res, expect)


def test_equal():
    """
    test equal
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            out_cond = fluid.data(name="input1", shape=[1], dtype='bool')
            label = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit1 = fluid.layers.assign(np.array([3, 3], dtype="int32"))
            label_cond = fluid.layers.assign(np.array([1, 2], dtype="int32"))

            out1 = fluid.layers.equal(x=label, y=limit)  # out1=[True, True]
            out2 = fluid.layers.equal(x=label, y=limit1)  # out2=[True, False]

            out3 = fluid.layers.equal(
                x=label_cond, y=limit,
                cond=out_cond)  # out2=[False, True] out_cond=[False, True]
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[out1, out2, out3])
            expect1 = np.array([True, True])
            expect2 = np.array([True, False])
            expect3 = np.array([False, True])
            tools.compare(res[0], expect1)
            tools.compare(res[1], expect2)
            tools.compare(res[2], expect3)


def test_greater_equal():
    """
    test greater_equal
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            out_cond = fluid.data(name="input1", shape=[1], dtype='bool')
            label = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit1 = fluid.layers.assign(np.array([3, 1], dtype="int32"))
            label_cond = fluid.layers.assign(np.array([1, 3], dtype="int32"))
            out1 = fluid.layers.greater_equal(
                x=label, y=limit)  # out1=[True, True]
            out2 = fluid.layers.greater_equal(
                x=label, y=limit1)  # out2=[True, True]
            out3 = fluid.layers.greater_equal(
                x=label_cond, y=limit, cond=out_cond)  # out3=[False, True]
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[out1, out2, out3])
            expect1 = np.array([True, True])
            expect2 = np.array([True, True])
            expect3 = np.array([False, True])
            tools.compare(res[0], expect1)
            tools.compare(res[1], expect2)
            tools.compare(res[2], expect3)


def test_greater_equal1():
    """
    test greater_equal cond=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            out_cond = fluid.data(name="input1", shape=[1], dtype='bool')
            label = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit1 = fluid.layers.assign(np.array([3, 1], dtype="int32"))
            label_cond = fluid.layers.assign(np.array([1, 3], dtype="int32"))
            out1 = fluid.layers.greater_equal(
                x=label, y=limit)  # out1=[True, True]
            out2 = fluid.layers.greater_equal(
                x=label, y=limit1)  # out2=[True, True]
            out3 = fluid.layers.greater_equal(
                x=label_cond, y=limit, cond=None)  # out3=[False, True]
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[out1, out2, out3])
            expect1 = np.array([True, True])
            expect2 = np.array([True, True])
            expect3 = np.array([False, True])
            tools.compare(res[0], expect1)
            tools.compare(res[1], expect2)
            tools.compare(res[2], expect3)


def test_greater_than():
    """
    test greater_than cond=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            out_cond = fluid.data(name="input1", shape=[1], dtype='bool')
            label = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit1 = fluid.layers.assign(np.array([3, 1], dtype="int32"))
            label_cond = fluid.layers.assign(np.array([1, 3], dtype="int32"))
            out1 = fluid.layers.greater_than(
                x=label, y=limit)  # out1=[False, False]
            out2 = fluid.layers.greater_than(
                x=label, y=limit1)  # out2=[False, True]
            out3 = fluid.layers.greater_than(
                x=label_cond, y=limit, cond=None)  # out3=[False, True]
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[out1, out2, out3])
            expect1 = np.array([False, False])
            expect2 = np.array([False, True])
            expect3 = np.array([False, True])
            tools.compare(res[0], expect1)
            tools.compare(res[1], expect2)
            tools.compare(res[2], expect3)


def test_greater_than1():
    """
    test greater_than
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            out_cond = fluid.data(name="input1", shape=[1], dtype='bool')
            label = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit1 = fluid.layers.assign(np.array([3, 1], dtype="int32"))
            label_cond = fluid.layers.assign(np.array([1, 3], dtype="int32"))
            out1 = fluid.layers.greater_than(
                x=label, y=limit)  # out1=[False, False]
            out2 = fluid.layers.greater_than(
                x=label, y=limit1)  # out2=[False, True]
            out3 = fluid.layers.greater_than(
                x=label_cond, y=limit, cond=out_cond)  # out3=[False, True]
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[out1, out2, out3])
            expect1 = np.array([False, False])
            expect2 = np.array([False, True])
            expect3 = np.array([False, True])
            tools.compare(res[0], expect1)
            tools.compare(res[1], expect2)
            tools.compare(res[2], expect3)


def test_ifelse():
    """
    test ifelse
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(
                name='x',
                shape=[4, 1],
                dtype='float32',
                append_batch_size=False)
            y = fluid.layers.data(
                name='y',
                shape=[4, 1],
                dtype='float32',
                append_batch_size=False)
            x_d = np.array([[3], [1], [-6], [-3]]).astype(np.float32)
            y_d = np.zeros((4, 1)).astype(np.float32)
            # 比较x, y对元素的大小，输出cond, cond是shape为[4, 1]，数据类型为bool的2-D tensor。
            # 根据输入数据x_d, y_d，可以推断出cond中的数据为[[true], [true], [false], [false]]
            cond = fluid.layers.greater_than(x, y)
            # 同其他常见OP不同的是，该OP返回的ie是一个IfElse OP的对象
            ie = fluid.layers.IfElse(cond)
            with ie.true_block():
                # 在这个block中，根据cond条件，获取x中对应条件为true维度的数据，并减去10
                out_1 = ie.input(x)
                out_1 = out_1 - 10
                ie.output(out_1)
            with ie.false_block():
                # 在这个block中，根据cond条件，获取x中对应条件为false维度的数据，并加上10
                out_1 = ie.input(x)
                out_1 = out_1 + 10
                ie.output(out_1)
            # 根据cond条件将两个block中处理后的数据进行合并，此处的output为输出，类型为List，List中的元素类型为Variable。
            output = ie()  # [array([[-7.], [-9.], [ 4.], [ 7.]], dtype=float32)]
            # 将输出List中的第一个Variable获取出来，并计算所有元素和
            out = fluid.layers.reduce_sum(output[0])
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            res = exe.run(fluid.default_main_program(),
                          feed={"x": x_d,
                                "y": y_d},
                          fetch_list=[out])
            tools.compare(res[0][0], -5)


def test_increment():
    """
    test increment
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            counter = fluid.layers.zeros(shape=[1], dtype='float32')  # [0.]
            x = fluid.layers.increment(counter)
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[x])[0][0]
            tools.compare(res, 1)


def test_increment1():
    """
    test increment value=33
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            counter = fluid.layers.zeros(shape=[1], dtype='float32')  # [0.]
            x = fluid.layers.increment(counter, value=33)
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[x])[0][0]
            tools.compare(res, 33)


def test_increment2():
    """
    test increment value=5 in_place = False
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            counter = fluid.layers.zeros(shape=[1], dtype='float32')  # [0.]
            x = fluid.layers.increment(counter, value=33, in_place=False)
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[x, counter])
            tools.compare(res, [[33], [0]])


def test_increment3():
    """
    test increment value=5 in_place = True
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            counter = fluid.layers.zeros(shape=[1], dtype='float32')  # [0.]
            x = fluid.layers.increment(counter, value=33, in_place=True)
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[x, counter])
            tools.compare(res, [[33], [33]])


def test_is_empty():
    """
    test is_empty
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            counter = fluid.layers.zeros(shape=[1], dtype='float32')  # [0.]
            x = fluid.layers.is_empty(counter)
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[x])[0][0]
            print(res)
            tools.compare(res, False)


def test_is_empty1():
    """
    test is_empty counter=None cond="SS"
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            counter = fluid.layers.data("a", shape=[10], dtype="float32")
            x = fluid.layers.is_empty(counter)
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog,
                          feed={"a": np.array([])},
                          fetch_list=[x])[0][0]
            print(res)
            tools.compare(res, True)


def test_less_equal():
    """
    test less_equal
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            out_cond = fluid.data(name="input1", shape=[1], dtype='bool')
            label = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit1 = fluid.layers.assign(np.array([3, 1], dtype="int32"))
            label_cond = fluid.layers.assign(np.array([1, 3], dtype="int32"))
            out1 = fluid.layers.less_equal(
                x=label, y=limit)  # out1=[True, True]
            out2 = fluid.layers.less_equal(
                x=label, y=limit1)  # out2=[True, False]
            out3 = fluid.layers.less_equal(
                x=label_cond, y=limit, cond=out_cond)  # out3=[True, False]
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[out1, out2, out_cond])
            expect1 = np.array([True, True])
            expect2 = np.array([True, False])
            expect3 = np.array([True, False])
            tools.compare(res[0], expect1)
            tools.compare(res[1], expect2)
            tools.compare(res[2], expect3)


def test_less_equal1():
    """
    test less_equal cond=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            out_cond = fluid.data(name="input1", shape=[1], dtype='bool')
            label = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit1 = fluid.layers.assign(np.array([3, 1], dtype="int32"))
            label_cond = fluid.layers.assign(np.array([1, 3], dtype="int32"))
            out1 = fluid.layers.less_equal(
                x=label, y=limit)  # out1=[True, True]
            out2 = fluid.layers.less_equal(
                x=label, y=limit1)  # out2=[True, False]
            out3 = fluid.layers.less_equal(
                x=label_cond, y=limit, cond=out_cond)  # out3=[True, False]
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[out1, out2, out3])
            expect1 = np.array([True, True])
            expect2 = np.array([True, False])
            expect3 = np.array([True, False])
            tools.compare(res[0], expect1)
            tools.compare(res[1], expect2)
            tools.compare(res[2], expect3)


def test_less_than():
    """
    test less_than
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            out_cond = fluid.data(name="input1", shape=[1], dtype='bool')
            label = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit1 = fluid.layers.assign(np.array([3, 1], dtype="int32"))
            label_cond = fluid.layers.assign(np.array([1, 3], dtype="int32"))
            out1 = fluid.layers.less_than(
                x=label, y=limit)  # out1=[False, False]
            out2 = fluid.layers.less_than(
                x=label, y=limit1)  # out2=[False, False]
            out3 = fluid.layers.less_than(
                x=label_cond, y=limit, cond=out_cond)  # out3=[True, False]
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[out1, out2, out_cond])
            expect1 = np.array([False, False])
            expect2 = np.array([False, False])
            expect3 = np.array([True, False])
            tools.compare(res[0], expect1)
            tools.compare(res[1], expect2)
            tools.compare(res[2], expect3)


def test_less_than1():
    """
    test less_than cond=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            out_cond = fluid.data(name="input1", shape=[1], dtype='bool')
            label = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit1 = fluid.layers.assign(np.array([3, 1], dtype="int32"))
            label_cond = fluid.layers.assign(np.array([1, 3], dtype="int32"))
            out1 = fluid.layers.less_than(
                x=label, y=limit)  # out1=[False, False]
            out2 = fluid.layers.less_than(
                x=label, y=limit1)  # out2=[False, False]
            out3 = fluid.layers.less_than(
                x=label_cond, y=limit)  # out3=[True, False]
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[out1, out2, out3])
            expect1 = np.array([False, False])
            expect2 = np.array([False, False])
            expect3 = np.array([True, False])
            tools.compare(res[0], expect1)
            tools.compare(res[1], expect2)
            tools.compare(res[2], expect3)


def test_not_equal():
    """
    test not_equal
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            out_cond = fluid.data(name="input1", shape=[1], dtype='bool')
            label = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit1 = fluid.layers.assign(np.array([3, 1], dtype="int32"))
            label_cond = fluid.layers.assign(np.array([1, 3], dtype="int32"))
            out1 = fluid.layers.not_equal(
                x=label, y=limit)  # out1=[False, False]
            out2 = fluid.layers.not_equal(
                x=label, y=limit1)  # out2=[False, True]
            out3 = fluid.layers.not_equal(
                x=label_cond, y=limit, cond=out_cond)  # out3=[True, True]
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[out1, out2, out_cond])
            expect1 = np.array([False, False])
            expect2 = np.array([False, True])
            expect3 = np.array([True, True])
            tools.compare(res[0], expect1)
            tools.compare(res[1], expect2)
            tools.compare(res[2], expect3)


def test_not_equal1():
    """
    test not_equal cond=None
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            out_cond = fluid.data(name="input1", shape=[1], dtype='bool')
            label = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            limit1 = fluid.layers.assign(np.array([3, 1], dtype="int32"))
            label_cond = fluid.layers.assign(np.array([1, 3], dtype="int32"))
            out1 = fluid.layers.not_equal(
                x=label, y=limit)  # out1=[False, False]
            out2 = fluid.layers.not_equal(
                x=label, y=limit1)  # out2=[False, True]
            out3 = fluid.layers.not_equal(
                x=label_cond, y=limit)  # out3=[True, True]
            place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, fetch_list=[out1, out2, out3])
            expect1 = np.array([False, False])
            expect2 = np.array([False, True])
            expect3 = np.array([True, True])
            tools.compare(res[0], expect1)
            tools.compare(res[1], expect2)
            tools.compare(res[2], expect3)
