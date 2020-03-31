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
"""test static tensor."""
import paddle.fluid as fluid
import numpy as np
import math
import tools


def test_argmax():
    """
    test argmax
    Returns:
        None
    axis = 0
    expect = [[0, 0, 0, 0],
     [1, 1, 1, 1],
     [0, 0, 0, 1]]
    """
    in1 = np.array([[[5, 8, 9, 5],
                     [0, 0, 1, 7],
                     [6, 9, 2, 4]],
                    [[5, 2, 4, 2],
                     [4, 7, 7, 9],
                     [1, 7, 0, 6]]])
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(in1)
        res = fluid.layers.argmax(x=x, axis=0)
        expect = [[0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [0, 0, 0, 1]]
        tools.compare(res.numpy(), expect)
        # [[2 3 1]
        #  [0 3 1]]
        # [[0 0 0 0]
        #  [1 1 1 1]
        #  [0 0 0 1]]
        # [[2 2 0 1]
        #  [0 1 1 1]]
        # [[2 3 1]
        #  [0 3 1]]


def test_argmax1():
    """
    test argmax1
    Returns:
        None
    axis = 1
    expect = [[2, 2, 0, 1],
                  [0, 1, 1, 1]]
    """
    in1 = np.array([[[5, 8, 9, 5],
                     [0, 0, 1, 7],
                     [6, 9, 2, 4]],
                    [[5, 2, 4, 2],
                     [4, 7, 7, 9],
                     [1, 7, 0, 6]]])
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(in1)
        res = fluid.layers.argmax(x=x, axis=1)
        expect = [[2, 2, 0, 1],
                  [0, 1, 1, 1]]
        tools.compare(res.numpy(), expect)


def test_argmax2():
    """

    Returns:
        None
    axis = 2
    expect = [[2, 3, 1],
                  [0, 3, 1]]
    """

    in1 = np.array([[[5, 8, 9, 5],
                     [0, 0, 1, 7],
                     [6, 9, 2, 4]],
                    [[5, 2, 4, 2],
                     [4, 7, 7, 9],
                     [1, 7, 0, 6]]])
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(in1)
        res = fluid.layers.argmax(x=x, axis=2)
        expect = [[2, 3, 1],
                  [0, 3, 1]]
        tools.compare(res.numpy(), expect)


def test_argmax3():
    """

    Returns:
        None
    axis = -1
    expect = [[2, 3, 1],
                  [0, 3, 1]]
    """

    in1 = np.array([[[5, 8, 9, 5],
                     [0, 0, 1, 7],
                     [6, 9, 2, 4]],
                    [[5, 2, 4, 2],
                     [4, 7, 7, 9],
                     [1, 7, 0, 6]]])
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(in1)
        res = fluid.layers.argmax(x=x, axis=-1)
        expect = [[2, 3, 1],
                  [0, 3, 1]]
        tools.compare(res.numpy(), expect)


def test_argmin():
    """
    test argmin axis=-1

    Returns:
        None

    """
    in1 = np.array([[[5, 8, 9, 5],
                     [0, 0, 1, 7],
                     [6, 9, 2, 4]],
                    [[5, 2, 4, 2],
                     [4, 7, 7, 9],
                     [1, 7, 0, 6]]])
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(in1)
        res = fluid.layers.argmin(x=x, axis=-1)
        expect = [[0, 0, 2], [1, 0, 2]]
        tools.compare(res.numpy(), expect)


def test_argmin1():
    """
    test argmin axis=0

    Returns:
        None

    """
    in1 = np.array([[[5, 8, 9, 5],
                     [0, 0, 1, 7],
                     [6, 9, 2, 4]],
                    [[5, 2, 4, 2],
                     [4, 7, 7, 9],
                     [1, 7, 0, 6]]])
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(in1)
        res = fluid.layers.argmin(x=x, axis=0)
        expect = [[0, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 0]]
        tools.compare(res.numpy(), expect)


def test_argmin2():
    """
    test argmin axis=1

    Returns:
        None

    """
    in1 = np.array([[[5, 8, 9, 5],
                     [0, 0, 1, 7],
                     [6, 9, 2, 4]],
                    [[5, 2, 4, 2],
                     [4, 7, 7, 9],
                     [1, 7, 0, 6]]])
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(in1)
        res = fluid.layers.argmin(x=x, axis=1)
        expect = [[1, 1, 1, 2], [2, 0, 2, 0]]
        tools.compare(res.numpy(), expect)


def test_argmin3():
    """
    test argmin axis=2

    Returns:
        None

    """
    in1 = np.array([[[5, 8, 9, 5],
                     [0, 0, 1, 7],
                     [6, 9, 2, 4]],
                    [[5, 2, 4, 2],
                     [4, 7, 7, 9],
                     [1, 7, 0, 6]]])
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(in1)
        res = fluid.layers.argmin(x=x, axis=2)
        expect = [[0, 0, 2], [1, 0, 2]]
        tools.compare(res.numpy(), expect)

def test_argsort():
    """
    test argsort axis=-1 descending=False

    Returns:
        None
    """
    in1 = np.array([[[5, 8, 9, 5],
                     [0, 0, 1, 7],
                     [6, 9, 2, 4]],
                    [[5, 2, 4, 2],
                     [4, 7, 7, 9],
                     [1, 7, 0, 6]]]).astype(np.float32)
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(in1)
        res = fluid.layers.argsort(input=x, axis=-1)  # same as axis==2
        expect = [[[5, 5, 8, 9], [0, 0, 1, 7], [2, 4, 6, 9]],
                  [[2, 2, 4, 5], [4, 7, 7, 9], [0, 1, 6, 7]]]
        index = [[[0, 3, 1, 2], [0, 1, 2, 3], [2, 3, 0, 1]],
                 [[1, 3, 2, 0], [0, 1, 2, 3], [2, 0, 3, 1]]]
        tools.compare(res[0].numpy(), expect)
        tools.compare(res[1].numpy(), index)


def test_argsort1():
    """
    test argsort axis=-1 descending=True

    Returns:
        None
    """
    in1 = np.array([[[5, 8, 9, 5],
                     [0, 0, 1, 7],
                     [6, 9, 2, 4]],
                    [[5, 2, 4, 2],
                     [4, 7, 7, 9],
                     [1, 7, 0, 6]]]).astype(np.float32)
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(in1)
        res = fluid.layers.argsort(input=x, axis=-1, descending=True)  # same as axis==2
        expect = [[[9, 8, 5, 5], [7, 1, 0, 0], [9, 6, 4, 2]],
                  [[5, 4, 2, 2], [9, 7, 7, 4], [7, 6, 1, 0]]]
        index = [[[2, 1, 0, 3], [3, 2, 0, 1], [1, 0, 3, 2]],
                 [[0, 2, 1, 3], [3, 1, 2, 0], [1, 3, 0, 2]]]
        tools.compare(res[0].numpy(), expect)
        tools.compare(res[1].numpy(), index)


def test_argsort2():
    """
    test argsort axis=0 descending=False

    Returns:
        None
    """
    in1 = np.array([[[5, 8, 9, 5],
                     [0, 0, 1, 7],
                     [6, 9, 2, 4]],
                    [[5, 2, 4, 2],
                     [4, 7, 7, 9],
                     [1, 7, 0, 6]]]).astype(np.float32)
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(in1)
        res = fluid.layers.argsort(input=x, axis=0, descending=False)
        expect = [[[5, 2, 4, 2], [0, 0, 1, 7], [1, 7, 0, 4]],
                  [[5, 8, 9, 5], [4, 7, 7, 9], [6, 9, 2, 6]]]
        tools.compare(res[0].numpy(), expect)


def test_argsort3():
    """
    test argsort axis=1 descending=False

    Returns:
        None
    """
    in1 = np.array([[[5, 8, 9, 5],
                     [0, 0, 1, 7],
                     [6, 9, 2, 4]],
                    [[5, 2, 4, 2],
                     [4, 7, 7, 9],
                     [1, 7, 0, 6]]]).astype(np.float32)
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(in1)
        res = fluid.layers.argsort(input=x, axis=1, descending=False)
        expect = [[[0, 0, 1, 4], [5, 8, 2, 5], [6, 9, 9, 7]],
                  [[1, 2, 0, 2], [4, 7, 4, 6], [5, 7, 7, 9]]]
        tools.compare(res[0].numpy(), expect)


def test_assign():
    """
    test assign

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.layers.fill_constant(shape=[3, 2], value=2.5, dtype='float64')
        result1 = fluid.layers.create_tensor(dtype='float64')
        fluid.layers.assign(data, result1)
        result2 = fluid.layers.assign(data)
        result3 = fluid.layers.assign(np.array([[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]],
                                               dtype='float32'))
        expect = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
        tools.compare(result1.numpy(), expect)
        tools.compare(result2.numpy(), expect)
        tools.compare(result3.numpy(), expect)


def test_cast():
    """
    test cast

    Returns:
        None
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            place = fluid.CPUPlace()
            x_lod = fluid.layers.data(name="x", shape=[1], lod_level=1)
            cast_res1 = fluid.layers.cast(x=x_lod, dtype="uint8")
            cast_res2 = fluid.layers.cast(x=x_lod, dtype=np.int32)
            exe = fluid.Executor(place)
            exe.run(startup_program)
            x_i_lod = fluid.core.LoDTensor()
            x_i_lod.set(np.array([[1.3, -2.4], [0, 4]]).astype("float32"), place)
            x_i_lod.set_recursive_sequence_lengths([[0, 2]])
            res1 = exe.run(train_program, feed={'x': x_i_lod}, fetch_list=[cast_res1], return_numpy=False)
            res2 = exe.run(train_program, feed={'x': x_i_lod}, fetch_list=[cast_res2], return_numpy=False)
            expect1 = [[1, 254], [0, 4]]
            assert np.array(res1[0]).dtype == np.uint8
            tools.compare(np.array(res1[0]), expect1)
            expect2 = [[1, -2], [0, 4]]
            assert np.array(res2[0]).dtype == np.int32
            tools.compare(np.array(res2[0]), expect2)


def test_concat():
    """
    test concat

    Returns:
        None
    """
    in1 = np.array([[1, 2, 3],
                    [4, 5, 6]])
    in2 = np.array([[11, 12, 13],
                    [14, 15, 16]])
    in3 = np.array([[21, 22],
                    [23, 24]])
    with fluid.dygraph.guard():
        x1 = fluid.dygraph.to_variable(in1)
        x2 = fluid.dygraph.to_variable(in2)
        x3 = fluid.dygraph.to_variable(in3)
        out1 = fluid.layers.concat(input=[x1, x2, x3], axis=-1)
        out2 = fluid.layers.concat(input=[x1, x2], axis=0)
        expect1 = [[1, 2, 3, 11, 12, 13, 21, 22], [4, 5, 6, 14, 15, 16, 23, 24]]
        tools.compare(out1.numpy(), expect1)
        expect2 = [[1, 2, 3], [4, 5, 6], [11, 12, 13], [14, 15 ,16]]
        tools.compare(out2.numpy(), expect2)


def test_create_global_var():
    """
    test create_global_var

    Returns:
        None
    """
    with fluid.dygraph.guard():
        var = fluid.layers.create_global_var(shape=[2, 3], value=1.0, dtype='float32',
                                       persistable=True, force_cpu=True, name='new_var')
        expect = [[1, 1, 1], [1, 1, 1]]
        tools.compare(var.numpy(), expect)


def test_diag():
    """
    test diag

    Returns:
        None
    """
    with fluid.dygraph.guard():
        diagonal = np.arange(3, 6, dtype='int32')
        data = fluid.layers.diag(diagonal)
        expect = [[3, 0, 0], [0, 4, 0], [0, 0, 5]]
        tools.compare(data.numpy(), expect)


def test_fill_constant():
    """
    test fill constant

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data1 = fluid.layers.fill_constant(shape=[2, 2], value=0, dtype='int64')
        expect1 = [[0, 0], [0, 0]]
        tools.compare(data1.numpy(), expect1)
        data2 = fluid.layers.fill_constant(shape=[2, 2], value=5, dtype='int64', out=data1)
        expect2 = [[5, 5], [5, 5]]
        tools.compare(data2.numpy(), expect2)


def test_fill_constant_batch_size_like():
    """
    test fill_constant_batch_size_like

    Returns:
        None

    """
    with fluid.dygraph.guard():
        like = fluid.layers.fill_constant(shape=[1, 2], value=10, dtype='int64')
        data = fluid.layers.fill_constant_batch_size_like(
            input=like, shape=[1], value=1, dtype='int64')
        expect = [1]
        tools.compare(data.numpy(), expect)


def test_fill_constant_batch_size_like1():
    """
    test fill_constant_batch_size_like force_cpu=True

    Returns:
        None

    """
    with fluid.dygraph.guard():
        like = fluid.layers.fill_constant(shape=[1, 2], value=10, dtype='int64')
        data = fluid.layers.fill_constant_batch_size_like(
            input=like, shape=[1], value=1, dtype='int64', force_cpu=True)
        expect = [1]
        tools.compare(data.numpy(), expect)


def test_has_inf():
    """
    test has_inf case:no inf

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(np.asarray([1, 2, 3]).astype("float32"))
        res = fluid.layers.has_inf(data).numpy()
        assert not res[0]


def test_has_inf1():
    """
    test has_inf case:has inf

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(np.asarray([1, np.inf, 3]))
        res = fluid.layers.has_inf(data).numpy()
        assert res[0]


def test_has_inf2():
    """
    test has_inf case:has nan

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(np.asarray([1, np.nan, 3]))
        res = fluid.layers.has_inf(data).numpy()
        assert not res[0]


def test_has_inf3():
    """
    test has_inf case:has inf && has nan

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(np.asarray([1, np.nan, np.inf]))
        res = fluid.layers.has_inf(data).numpy()
        assert res[0]


def test_has_nan():
    """
    test has_nan case:no nan

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(np.asarray([1, 2, 3]).astype("float32"))
        res = fluid.layers.has_nan(data).numpy()
        assert not res[0]


def test_has_nan1():
    """
    test has_nan case:has nan

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(np.asarray([1, np.nan, 3]))
        res = fluid.layers.has_nan(data).numpy()
        assert res[0]


def test_has_nan2():
    """
    test has_nan case:has inf

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(np.asarray([1, np.inf, 3]))
        res = fluid.layers.has_nan(data).numpy()
        assert not res[0]


def test_has_nan3():
    """
    test has_nan case:has inf && has nan

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(np.asarray([1, np.nan, np.inf]))
        res = fluid.layers.has_nan(data).numpy()
        assert res[0]


def test_isfinite():
    """
    test isfinite case: no isfinite

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(np.asarray([1, 2, 3]).astype("float32"))
        res = fluid.layers.isfinite(data).numpy()
        assert res[0]


def test_isfinite1():
    """
    test isfinite case: has inf

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(np.asarray([1, np.inf, 3]))
        res = fluid.layers.isfinite(data).numpy()
        assert not res[0]


def test_isfinite2():
    """
    test isfinite case: has nan

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(np.asarray([1, np.nan, 3]))
        res = fluid.layers.isfinite(data).numpy()
        assert not res[0]


def test_isfinite3():
    """
    test isfinite case: has nan && has inf

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(np.asarray([1, np.nan, np.inf]))
        res = fluid.layers.isfinite(data).numpy()
        assert not res[0]


def test_linspace():
    """
    test linspace

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data1 = fluid.layers.linspace(0, 10, 5, 'float32')
        data2 = fluid.layers.linspace(0, 10, 1, 'float64')
        data3 = fluid.layers.linspace(0, 12.5, 6, 'float32')
        expect1 = [0.0, 2.5, 5.0, 7.5, 10.0]
        expect2 = [0.0]
        expect3 = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]
        tools.compare(data1.numpy(), expect1)
        tools.compare(data2.numpy(), expect2)
        tools.compare(data3.numpy(), expect3)


def test_ones():
    """
    test ones

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.layers.ones(shape=[2, 4], dtype='float32')
        expect = [[1, 1, 1, 1], [1, 1, 1, 1]]
        tools.compare(data.numpy(), expect)


def test_range():
    """
    test range

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data1 = fluid.layers.range(0, 10, 2, 'int32')
        expect1 = [0, 2, 4, 6, 8]
        data2 = fluid.layers.range(0, 11, 1.5, 'float32')
        expect2 = [0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5]
        tools.compare(data1.numpy(), expect1)
        tools.compare(data2.numpy(), expect2)


def test_reverse():
    """
    test reverse

    Returns:
        None
    """
    with fluid.dygraph.guard():
        data = fluid.layers.assign(
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype='float32'))
        result1 = fluid.layers.reverse(data, 0)
        result2 = fluid.layers.reverse(data, [0, 1])
        expect1 = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        expect2 = [[8, 7, 6], [5, 4, 3], [2, 1, 0]]
        tools.compare(result1.numpy(), expect1)
        tools.compare(result2.numpy(), expect2)


def test_sums():
    """
    test sums

    Returns:
        None
    """
    with fluid.dygraph.guard():
        x0 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=1)
        x1 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=2)
        x2 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=3)
        x3 = fluid.layers.fill_constant(shape=[16, 32], dtype='int64', value=1)
        sum0 = fluid.layers.sums([x0, x1, x2])
        expect = x3.numpy() * 6
        tools.compare(sum0.numpy(), expect)


def tensor_array_to_tensor():
    """
    test tensor_array_to_tensor

    Returns:
        None
    """
    pass
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            a = np.array([[0.6, 0.1, 0.3],
                           [0.5, 0.3, 0.2]])
            b = np.array([[1.3], [1.8]])
            c = np.array([[2.3, 2.1],
                           [2.5, 2.4]])
            A = fluid.LoDTensor()
            B = fluid.LoDTensor()
            C = fluid.LoDTensor()
            res = fluid.layers.tensor_array_to_tensor([A, B, C], axis=1, use_stack=False)
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(train_program)
            res = exe.run(compiled_prog, feed={A: a, B: b, C: c}, fetch_list=[res])
            print(res)


def test_zeros():
    """
    test zeros

    Returns:
        None
    """
    with fluid.dygraph.guard():
        res = fluid.layers.zeros(shape=[3, 2], dtype='float32')
        expect = [[0, 0], [0, 0], [0, 0]]
        tools.compare(res.numpy(), expect)


def test_zeros_like():
    """
    test zeros_like

    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(np.array([[1, 2], [4, 0], [3, 3]]))
        res = fluid.layers.zeros_like(x)
        expect = [[0, 0], [0, 0], [0, 0]]
        tools.compare(res.numpy(), expect)


