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
"""nn test cases"""
import paddle.fluid as fluid
import paddle
import numpy as np
import math
import tools
import platform
from paddle.fluid.dygraph.base import to_variable


def test_L1Loss():
    """
    test L1 loss reduction=none
    Returns:
        None
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(main_program=main_program, startup_program=startup_program):
            l1_loss = paddle.nn.loss.L1Loss("none")
            input = fluid.data(name="input", shape=[3, 3])
            label = fluid.data(name="label", shape=[3, 3])
            output = l1_loss(input, label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            input_data = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]).astype("float32")
            label_data = np.array([[2, 3, 2], [3, 1, 2], [1, 1, 1]]).astype("float32")
            output_data = exe.run(fluid.default_main_program(),
                                  feed={"input": input_data, "label": label_data},
                                  fetch_list=[output],
                                  return_numpy=True)
            expect = np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
            tools.compare(output_data[0], expect)


def test_L1Loss1():
    """
    test L1 loss reduction=sum
    Returns:
        None
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(main_program=main_program, startup_program=startup_program):
            l1_loss = paddle.nn.loss.L1Loss("sum")
            input = fluid.data(name="input", shape=[3, 3])
            label = fluid.data(name="label", shape=[3, 3])
            output = l1_loss(input, label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            input_data = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]).astype("float32")
            label_data = np.array([[2, 3, 2], [3, 1, 2], [1, 1, 1]]).astype("float32")
            output_data = exe.run(fluid.default_main_program(),
                                  feed={"input": input_data, "label": label_data},
                                  fetch_list=[output],
                                  return_numpy=True)
            expect = np.array([6], dtype=np.float32)
            tools.compare(output_data[0], expect)


def test_L1Loss2():
    """
    test L1 loss reduction=mean
    Returns:
        None
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(main_program=main_program, startup_program=startup_program):
            l1_loss = paddle.nn.loss.L1Loss("mean")
            input = fluid.data(name="input", shape=[3, 3])
            label = fluid.data(name="label", shape=[3, 3])
            output = l1_loss(input, label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            input_data = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]).astype("float32")
            label_data = np.array([[2, 3, 2], [3, 1, 2], [1, 1, 1]]).astype("float32")
            output_data = exe.run(fluid.default_main_program(),
                                  feed={"input": input_data, "label": label_data},
                                  fetch_list=[output],
                                  return_numpy=True)
            expect = np.array([0.66666666], dtype=np.float32)
            tools.compare(output_data[0], expect)


def test_L1Loss3():
    """
    test L1 loss type = int32
    Returns:
        None
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(main_program=main_program, startup_program=startup_program):
            l1_loss = paddle.nn.loss.L1Loss("sum")
            input = fluid.data(name="input", shape=[3, 3], dtype="int32")
            label = fluid.data(name="label", shape=[3, 3], dtype="int32")
            output = l1_loss(input, label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            input_data = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]).astype("int32")
            label_data = np.array([[2, 3, 2], [3, 1, 2], [1, 1, 1]]).astype("int32")
            output_data = exe.run(fluid.default_main_program(),
                                  feed={"input": input_data, "label": label_data},
                                  fetch_list=[output],
                                  return_numpy=True)
            expect = np.array([6], dtype=np.int32)
            tools.compare(output_data[0], expect)


def test_L1Loss4():
    """
    test L1 loss type = int64
    Returns:
        None
    """

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(main_program=main_program, startup_program=startup_program):
            l1_loss = paddle.nn.loss.L1Loss("mean")
            input = fluid.data(name="input", shape=[3, 3], dtype="int64")
            label = fluid.data(name="label", shape=[3, 3], dtype="int64")
            output = l1_loss(input, label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            input_data = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]).astype("int64")
            label_data = np.array([[2, 3, 2], [3, 1, 2], [1, 1, 1]]).astype("int64")
            output_data = exe.run(fluid.default_main_program(),
                                  feed={"input": input_data, "label": label_data},
                                  fetch_list=[output],
                                  return_numpy=True)
            expect = np.array([0], dtype=np.int64)
            tools.compare(output_data[0], expect)


def test_L1Loss5():
    """
    test L1 loss type = float64
    Returns:
        None
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(main_program=main_program, startup_program=startup_program):
            l1_loss = paddle.nn.loss.L1Loss("mean")
            input = fluid.data(name="input", shape=[3, 3], dtype="float64")
            label = fluid.data(name="label", shape=[3, 3], dtype="float64")
            output = l1_loss(input, label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            input_data = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]).astype("float64")
            label_data = np.array([[2, 3, 2], [3, 1, 2], [1, 1, 1]]).astype("float64")
            output_data = exe.run(fluid.default_main_program(),
                                  feed={"input": input_data, "label": label_data},
                                  fetch_list=[output],
                                  return_numpy=True)
            expect = np.array([0.66666666], dtype=np.float64)
            tools.compare(output_data[0], expect)


def __allclose__(input, other, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):
    """
    compute allclose
    Args:
        input: input tensor
        other: other tensor
        rtol: relative tolerance
        atol: absolute tolerance
        equal_nan: if true ,two nans will be equal
        name: name

    Returns:
        Boolean
    """
    arr = abs(input - other) <= atol + rtol * abs(other)
    if False in arr:
        return [False]
    else:
        return [True]


def test_allclose():
    """
    test allclose
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = to_variable(np.array([100, 200]).astype("float32"))
        other = to_variable(np.array([100, 2000]).astype("float32"))
        res = paddle.tensor.logic.allclose(input, other)
        expect = __allclose__(np.array([100, 200]).astype("float32"), np.array([100, 2000]).astype("float32"))
        tools.compare(res.numpy(), expect)


def test_allclose1():
    """
    test allclose
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = to_variable(np.array([100, 200]).astype("float32"))
        other = to_variable(np.array([100, 200]).astype("float32"))
        res = paddle.tensor.logic.allclose(input, other)
        expect = __allclose__(np.array([100, 200]).astype("float32"), np.array([100, 200]).astype("float32"))
        tools.compare(res.numpy(), expect)


def test_allclose2():
    """
    test allclose type=float64
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = to_variable(np.array([100, 200]).astype("float64"))
        other = to_variable(np.array([100, 2000]).astype("float64"))
        res = paddle.tensor.logic.allclose(input, other)
        expect = __allclose__(np.array([100, 200]).astype("float64"), np.array([100, 2000]).astype("float64"))
        tools.compare(res.numpy(), expect)


def test_allclose3():
    """
    test allclose type=float64
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = to_variable(np.array([100, 200]).astype("float64"))
        other = to_variable(np.array([100, 200]).astype("float64"))
        res = paddle.tensor.logic.allclose(input, other)
        expect = __allclose__(np.array([100, 200]).astype("float64"), np.array([100, 200]).astype("float64"))
        tools.compare(res.numpy(), expect)


def test_allclose4():
    """
    test allclose type=float64 rtol=0
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = to_variable(np.array([100, 1e-7]).astype("float64"))
        other = to_variable(np.array([100, 1e-8]).astype("float64"))
        res = paddle.tensor.logic.allclose(input, other, rtol=0)
        expect = __allclose__(np.array([100, 1e-7]).astype("float64"), np.array([100, 1e-8]).astype("float64"),
                              rtol=0)
        tools.compare(res.numpy(), expect)


def test_allclose5():
    """
    test allclose type=float64 rtol=0
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = to_variable(np.array([100, 1e-9]).astype("float64"))
        other = to_variable(np.array([100, 1e-8]).astype("float64"))
        res = paddle.tensor.logic.allclose(input, other, rtol=0)
        expect = __allclose__(np.array([100, 1e-9]).astype("float64"), np.array([100, 1e-8]).astype("float64"),
                              rtol=0)
        tools.compare(res.numpy(), expect)


def test_allclose6():
    """
    test allclose type=float64 atol=0
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = to_variable(np.array([1+1e-8, 1e-8]).astype("float64"))
        other = to_variable(np.array([1, 1e-8]).astype("float64"))
        res = paddle.tensor.logic.allclose(input, other, atol=0)
        expect = __allclose__(np.array([1+1e-8, 1e-8]).astype("float64"), np.array([1, 1e-8]).astype("float64"),
                              atol=0)
        tools.compare(res.numpy(), expect)


def test_allclose7():
    """
    test allclose type=float64 atol=0
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = to_variable(np.array([1+1e-4, 1e-8]).astype("float64"))
        other = to_variable(np.array([1, 1e-8]).astype("float64"))
        res = paddle.tensor.logic.allclose(input, other, atol=0)
        expect = __allclose__(np.array([1+1e-4, 1e-8]).astype("float64"), np.array([1, 1e-8]).astype("float64"),
                              atol=0)
        tools.compare(res.numpy(), expect)


def test_allclose8():
    """
    test allclose type=float64 equal_nan=False
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = to_variable(np.array([math.nan, 1e-8]).astype("float64"))
        other = to_variable(np.array([math.nan, 1e-8]).astype("float64"))
        res = paddle.tensor.logic.allclose(input, other, equal_nan=False)
        tools.compare(res.numpy(), [False])


def test_allclose9():
    """
    test allclose type=float64 equal_nan=True
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = to_variable(np.array([math.nan, 1e-8]).astype("float64"))
        other = to_variable(np.array([math.nan, 1e-8]).astype("float64"))
        res = paddle.tensor.logic.allclose(input, other, equal_nan=True)
        tools.compare(res.numpy(), [True])


def test_allclose10():
    """
    test allclose type=float64 name=ss
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = to_variable(np.array([math.nan, 1e-8]).astype("float64"))
        other = to_variable(np.array([math.nan, 1e-8]).astype("float64"))
        res = paddle.tensor.logic.allclose(input, other, equal_nan=True, name="ss")
        tools.compare(res.numpy(), [True])


def test_dot():
    """
    test dot
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.array([1, 2, 3, 4, 5]).astype(np.float64)
        y = np.array([1, 2, 3, 4, 5]).astype(np.float64)
        x = to_variable(x)
        y = to_variable(y)
        res = paddle.dot(x, y)
        expect = [55]
        tools.compare(res.numpy(), expect)


def test_dot1():
    """
    test dot dtype=float32
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
        x = to_variable(x)
        y = to_variable(y)
        res = paddle.dot(x, y)
        expect = [55]
        tools.compare(res.numpy(), expect)


def test_dot2():
    """
    test dot dtype=int32
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.int32)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.int32)
        x = to_variable(x)
        y = to_variable(y)
        res = paddle.dot(x, y)
        expect = [55.0]
        tools.compare(res.numpy(), expect)


def test_dot3():
    """
    test dot dtype=int64
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.array([1, 2, 3, 4, 5]).astype(np.int64)
        y = np.array([1, 2, 3, 4, 5]).astype(np.int64)
        x = to_variable(x)
        y = to_variable(y)
        res = paddle.dot(x, y)
        expect = [55.0]
        tools.compare(res.numpy(), expect)


def test_dot4():
    """
    test dot dtype=int64 name=ss
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.array([1, 2, 3, 4, 5]).astype(np.int64)
        y = np.array([1, 2, 3, 4, 5]).astype(np.int64)
        x = to_variable(x)
        y = to_variable(y)
        res = paddle.dot(x, y, name="ss")
        expect = [55.0]
        tools.compare(res.numpy(), expect)


def math_logsumexp(data):
    """
    achieve logsumexp by numpy
    Args:
        data: float array

    Returns:
        Float
    """
    res = []
    for i in data:
        res.append(math.exp(i))
    return math.log(sum(res))


def test_logsumexp():
    """
    test logsumexp
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(np.array([1, 2, 3, 4, 5]).astype(np.float32))
        res = paddle.logsumexp(x).numpy()
        expect = math_logsumexp([1, 2, 3, 4, 5])
        tools.compare(res[0], expect, 1e-7)


def test_logsumexp1():
    """
    test logsumexp dim=-1
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(np.array([[2, 2], [3, 3]]).astype(np.float32))
        res = paddle.logsumexp(x, dim=-1).numpy()
        expect = [2.6931472, 3.6931472]
        tools.compare(res, expect, 1e-7)


def test_logsumexp2():
    """
    test logsumexp dim=0
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(np.array([[2, 2], [3, 3]]).astype(np.float32))
        res = paddle.logsumexp(x, dim=0).numpy()
        expect = [3.3132617, 3.3132617]
        tools.compare(res, expect, 1e-7)


def test_logsumexp3():
    """
    test logsumexp dim=1
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(np.array([[2, 2], [3, 3]]).astype(np.float32))
        res = paddle.logsumexp(x, dim=1).numpy()
        print(res)
        expect = [2.6931472, 3.6931472]
        tools.compare(res, expect, 1e-7)


def test_logsumexp4():
    """
    test logsumexp keep_dim=True
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(np.array([[2, 2], [3, 3]]).astype(np.float32))
        res = paddle.logsumexp(x, dim=1, keepdim=True).numpy()
        expect = [[2.6931472], [3.6931472]]
        tools.compare(res, expect, 1e-7)


def test_logsumexp5():
    """
    test logsumexp keep_dim=True
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(np.array([[2, 2], [3, 3]]).astype(np.float32))
        res = paddle.logsumexp(x, dim=None, keepdim=True).numpy()
        expect = [[4.0064087]]
        tools.compare(res, expect, 1e-7)


def test_logsumexp6():
    """
    test logsumexp keep_dim=True largedim
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(np.array([[[2, 2], [3, 3]], [[2, 2], [3, 3]]]).astype(np.float32))
        res = paddle.logsumexp(x, dim=1, keepdim=True).numpy()
        expect = [[[3.3132617, 3.3132617]], [[3.3132617, 3.3132617]]]
        tools.compare(res, expect, 1e-7)


def test_logsumexp7():
    """
    test logsumexp keep_dim=False largedim
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(np.array([[[2, 2], [3, 3]], [[2, 2], [3, 3]]]).astype(np.float32))
        res = paddle.logsumexp(x, dim=1, keepdim=False).numpy()
        expect = [[3.3132617, 3.3132617], [3.3132617, 3.3132617]]
        tools.compare(res, expect, 1e-7)


def test_logsumexp8():
    """
    test logsumexp keep_dim=True largedim dim=[0, 1, 2]
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(np.array([[[2, 2], [3, 3]], [[2, 2], [3, 3]]]).astype(np.float32))
        res = paddle.logsumexp(x, dim=[1, 2], keepdim=False).numpy()
        expect = [4.0064087, 4.0064087]
        tools.compare(res, expect, 1e-7)


def test_full():
    """
    test full default value
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = paddle.full(shape=[1000, 1000], fill_value=3.3, dtype=np.float32)
        expect = np.ones(shape=[1000, 1000]).astype(np.float32) * 3.3
        tools.compare(x.numpy(), expect)


def test_full1():
    """
    test full different dtype
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = paddle.fill_constant(shape=[1000, 1000], value=3.3, dtype=np.float16)
        print(x.numpy())
        x1 = paddle.full(shape=[1000, 1000], fill_value=3.3, dtype=np.float16)
        expect1 = (np.ones(shape=[1000, 1000]) * 3.3).astype(np.float16)
        x2 = paddle.full(shape=[1000, 1000], fill_value=3.3, dtype=np.float64)
        expect2 = (np.ones(shape=[1000, 1000]) * 3.3).astype(np.float64)
        x3 = paddle.full(shape=[1000, 1000], fill_value=3.3, dtype=np.int32)
        expect3 = (np.ones(shape=[1000, 1000]) * 3.3).astype(np.int32)
        x4 = paddle.full(shape=[1000, 1000], fill_value=3.3, dtype=np.int64)
        expect4 = (np.ones(shape=[1000, 1000]) * 3.3).astype(np.int64)
        x5 = paddle.full(shape=[1000, 1000], fill_value=3.3, dtype=np.bool)
        expect5 = (np.ones(shape=[1000, 1000]) * 3.3).astype(np.bool)
        tools.compare(x1.numpy(), expect1, delta=0.01)
        tools.compare(x2.numpy(), expect2)
        tools.compare(x3.numpy(), expect3)
        tools.compare(x4.numpy(), expect4)
        tools.compare(x5.numpy(), expect5)


def test_full2():
    """
    test full device = cpu
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = paddle.full(shape=[1000, 1000], fill_value=3.3, dtype=np.float32, device="cpu")
        expect = np.ones(shape=[1000, 1000]).astype(np.float32) * 3.3
        tools.compare(x.numpy(), expect)


def test_full3():
    """
    test full name=ss
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = paddle.full(shape=[1000, 1000], fill_value=3.3, dtype=np.float32, name="ss")
        expect = np.ones(shape=[1000, 1000]).astype(np.float32) * 3.3
        tools.compare(x.numpy(), expect)


def test_full4():
    """
    test full device = cpu out=a
    Returns:
        None
    """
    with fluid.dygraph.guard():
        a = to_variable(np.ones(shape=[1000, 1000]).astype(np.float32))
        x = paddle.full(out=a, shape=[1000, 1000], fill_value=3.3, dtype=np.float32, device="cpu")
        expect = np.ones(shape=[1000, 1000]).astype(np.float32) * 3.3
        tools.compare(a.numpy(), expect)


def test_zeros_like():
    """
    test zeros_like
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.zeros_like(x)
        expect = np.zeros(shape=[100, 100, 100])
        tools.compare(res.numpy(), expect)


def test_zeros_like1():
    """
    test zeros_like different dtype
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.zeros_like(x, dtype=np.bool)
        expect = np.zeros(shape=[100, 100, 100]).astype(np.bool)
        tools.compare(res.numpy(), expect)
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.zeros_like(x, dtype=np.float32)
        expect = np.zeros(shape=[100, 100, 100]).astype(np.float32)
        tools.compare(res.numpy(), expect)
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.zeros_like(x, dtype=np.float64)
        expect = np.zeros(shape=[100, 100, 100]).astype(np.float64)
        tools.compare(res.numpy(), expect)
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.zeros_like(x, dtype=np.int32)
        expect = np.zeros(shape=[100, 100, 100]).astype(np.int32)
        tools.compare(res.numpy(), expect)
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.zeros_like(x, dtype=np.int64)
        expect = np.zeros(shape=[100, 100, 100]).astype(np.int64)
        tools.compare(res.numpy(), expect)


def test_zeros_like2():
    """
    test zeros_like device=cpu
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.zeros_like(x, device="cpu")
        expect = np.zeros(shape=[100, 100, 100])
        tools.compare(res.numpy(), expect)


def test_zeros_like3():
    """
    test zeros_like device=cpu name=ss
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.zeros_like(x, device="cpu", name="ss")
        expect = np.zeros(shape=[100, 100, 100])
        tools.compare(res.numpy(), expect)


def test_ones_like():
    """
    test ones_like
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.ones_like(x)
        expect = np.ones(shape=[100, 100, 100])
        tools.compare(res.numpy(), expect)


def test_ones_like1():
    """
    test ones_like different dtype
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.ones_like(x, dtype=np.bool)
        expect = np.ones(shape=[100, 100, 100]).astype(np.bool)
        tools.compare(res.numpy(), expect)
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.ones_like(x, dtype=np.float32)
        expect = np.ones(shape=[100, 100, 100]).astype(np.float32)
        tools.compare(res.numpy(), expect)
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.ones_like(x, dtype=np.float64)
        expect = np.ones(shape=[100, 100, 100]).astype(np.float64)
        tools.compare(res.numpy(), expect)
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.ones_like(x, dtype=np.int32)
        expect = np.ones(shape=[100, 100, 100]).astype(np.int32)
        tools.compare(res.numpy(), expect)
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.ones_like(x, dtype=np.int64)
        expect = np.ones(shape=[100, 100, 100]).astype(np.int64)
        tools.compare(res.numpy(), expect)


def test_ones_like2():
    """
    test ones_like device=cpu
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.ones_like(x, device="cpu")
        expect = np.ones(shape=[100, 100, 100])
        tools.compare(res.numpy(), expect)


def test_ones_like3():
    """
    test ones_like device=cpu name=ss
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = to_variable(np.random.random(size=[100, 100, 100]))
        res = paddle.ones_like(x, device="cpu", name="ss")
        expect = np.ones(shape=[100, 100, 100])
        tools.compare(res.numpy(), expect)


def test_elementwise_equal():
    """
    test elementwise equal
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 4])
        paddle_x = to_variable(x)
        paddle_y = to_variable(y)
        res = paddle.elementwise_equal(paddle_x, paddle_y)
        expect = [1, 1, 1, 1, 0]
        tools.compare(res.numpy(), expect)


def test_elementwise_equal1():
    """
    test elementwise equal 2 dig
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([[1, 1], [2, 1], [1, 3]])
        paddle_x = to_variable(x)
        paddle_y = to_variable(y)
        res = paddle.elementwise_equal(paddle_x, paddle_y)
        expect = [[1, 1], [1, 0], [0, 1]]
        tools.compare(res.numpy(), expect)


def test_elementwise_equal2():
    """
    test elementwise equal name = ss
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([[1, 1], [2, 1], [1, 3]])
        paddle_x = to_variable(x)
        paddle_y = to_variable(y)
        res = paddle.elementwise_equal(paddle_x, paddle_y, name="ss")
        expect = [[1, 1], [1, 0], [0, 1]]
        tools.compare(res.numpy(), expect)


def test_randint():
    """
    test randint
    Returns:
        None
    """
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x = paddle.tensor.randint(low=1, high=5, shape=[3, 3], seed=33)
        if platform.system() == "Darwin":
            expect = [[3, 4, 1], [2, 4, 1], [1, 2, 3]]
        elif platform.system() == "Linux":
            expect = [[1, 4, 4], [2, 4, 2], [4, 1, 3]]
        else:
            expect = [[3, 2, 1], [2, 1, 2], [3, 3, 4]]
        tools.compare(x.numpy(), expect)


def test_randint1():
    """
    test randint high=None
    Returns:
        None
    """
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x = paddle.tensor.randint(low=1, shape=[3, 3])
        expect = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        tools.compare(x.numpy(), expect)


def test_randint2():
    """
    test randint device="cpu", name="ss"
    Returns:
        None
    """
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x = paddle.tensor.randint(low=1, high=5, shape=[3, 3], seed=33, device="cpu", name="ss")
        if platform.system() == "Darwin":
            expect = [[3, 4, 1], [2, 4, 1], [1, 2, 3]]
        elif platform.system() == "Linux":
            expect = [[1, 4, 4], [2, 4, 2], [4, 1, 3]]
        else:
            expect = [[3, 2, 1], [2, 1, 2], [3, 3, 4]]
        tools.compare(x.numpy(), expect)


def test_randint3():
    """
    test randint device="cpu", name="ss" out=a
    Returns:
        None
    """
    with fluid.dygraph.guard(fluid.CPUPlace()):
        a = to_variable(np.ones(shape=[3, 3]).astype(np.int32))
        x = paddle.tensor.randint(out=a, low=1, high=5, shape=[3, 3], seed=33, device="cpu", name="ss")
        if platform.system() == "Darwin":
            expect = [[3, 4, 1], [2, 4, 1], [1, 2, 3]]
        elif platform.system() == "Linux":
            expect = [[1, 4, 4], [2, 4, 2], [4, 1, 3]]
        else:
            expect = [[3, 2, 1], [2, 1, 2], [3, 3, 4]]
        tools.compare(a.numpy(), expect)


def test_manual_seed():
    """
    manual seed
    Returns:
        None
    """
    prog1 = fluid.default_startup_program()
    prog2 = fluid.Program()
    tools.compare(prog1.random_seed, 0)
    tools.compare(prog2.random_seed, 0)
    print(prog1.random_seed)
    print(prog2.random_seed)
    paddle.manual_seed(33)
    prog3 = fluid.Program()
    # default prog会被修改
    tools.compare(prog1.random_seed, 33)
    # 自定义的不会被修改
    tools.compare(prog2.random_seed, 0)
    tools.compare(prog3.random_seed, 33)


def test_diag_embed():
    """
    test diag embed default
    Returns:
        None
    """
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x = np.array([[1, 2, 3], [1, 2, 3]])
        res = paddle.nn.functional.diag_embed(to_variable(x))
        expect = [[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                  [[1, 0, 0], [0, 2, 0], [0, 0, 3]]]
        tools.compare(res.numpy(), expect)

def test_diag_embed1():
    """
    test diag embed offset=1
    Returns:
        None
    """
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x = np.array([[1, 2, 3], [1, 2, 3]])
        res = paddle.nn.functional.diag_embed(to_variable(x), offset=1)
        expect = [[[0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3], [0, 0, 0, 0]],
                  [[0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3], [0, 0, 0, 0]]]
        tools.compare(res.numpy(), expect)

def test_diag_embed2():
    """
    test diag embed dim1=0, dim2=1
    Returns:
        None
    """
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x = np.array([[1, 2, 3], [1, 2, 3]])
        res = paddle.nn.functional.diag_embed(to_variable(x), dim1=0, dim2=1)
        expect = [[[1, 1], [0, 0], [0, 0]],
                  [[0, 0], [2, 2], [0, 0]],
                  [[0, 0], [0, 0], [3, 3]]]
        tools.compare(res.numpy(), expect)


def test_diag_embed3():
    """
    test diag embed dim1=0, dim2=2
    Returns:
        None
    """
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x = np.array([[1, 2, 3], [1, 2, 3]])
        res = paddle.nn.functional.diag_embed(to_variable(x), dim1=0, dim2=2)
        expect = [[[1, 0, 0], [1, 0, 0]],
                  [[0, 2, 0], [0, 2, 0]],
                  [[0, 0, 3], [0, 0, 3]]]
        tools.compare(res.numpy(), expect)


def test_nn_relu():
    """
    test nn.relu
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x_np = np.random.uniform(-1, 1, [10, 12, 128, 128]).astype('float32')
        x = fluid.dygraph.to_variable(x_np)
        my_relu=paddle.nn.ReLU()
        out = my_relu(x)
        arr = []
        for i in x_np.flatten():
            if i < 0:
                arr.append(0)
            else:
                arr.append(i)
        expect = np.array(arr).reshape(10, 12, 128, 128)
        tools.compare(out.numpy(), expect)


def test_elementwise_sum():
    """
    test elementwise_sum
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([[1, 1], [2, 1], [1, 3]])
        paddle_x = to_variable(x)
        paddle_y = to_variable(y)
        res = paddle.elementwise_sum([paddle_x, paddle_y], name="ss")
        expect = [[2, 2], [4, 3], [4, 6]]
        tools.compare(res.numpy(), expect)


def test_matmul():
    """
    test matmul
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([1, 2, 3]).astype(np.float32)
        paddle_x = to_variable(x)
        paddle_y = to_variable(y)
        res = paddle.matmul(paddle_x, paddle_y, name="ss")
        expect = [14]
        tools.compare(res.numpy(), expect)
        print(res.numpy())


def test_matmul1():
    """
    test matmul alpha=2
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([1, 2, 3]).astype(np.float32)
        paddle_x = to_variable(x)
        paddle_y = to_variable(y)
        res = paddle.matmul(paddle_x, paddle_y, alpha=2, name="ss")
        expect = [28]
        tools.compare(res.numpy(), expect)
        print(res.numpy())


def test_matmul2():
    """
    test matmul
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([[1, 2, 3], [1, 2, 3]]).astype(np.float32)
        y = np.array([[1, 2, 3], [1, 2, 3]]).astype(np.float32)
        paddle_x = to_variable(x)
        paddle_y = to_variable(y)
        res = paddle.matmul(paddle_x, paddle_y, transpose_y=True, alpha=1, name="ss")
        expect = [[14, 14], [14, 14]]
        tools.compare(res.numpy(), expect)


def test_logsoftmax():
    """
    test logsoftmax
    Returns:
        None
    """
    data = np.array([[[-2.0, 3.0, -4.0, 5.0],
                      [3.0, -4.0, 5.0, -6.0],
                      [-7.0, -8.0, 8.0, 9.0]],
                     [[1.0, -2.0, -3.0, 4.0],
                      [-5.0, 6.0, 7.0, -8.0],
                      [6.0, 7.0, 8.0, 9.0]]]).astype('float32')
    my_log_softnmax = paddle.nn.LogSoftmax()
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(data)
        res = my_log_softnmax(data)
        expect = [[[ -7.1278,  -2.1278,  -9.1278,  -0.1278],
         [ -2.1271,  -9.1271,  -0.1271, -11.1271],
         [-16.3133, -17.3133,  -1.3133,  -0.3133]],

        [[ -3.0518,  -6.0518,  -7.0518,  -0.0518],
         [-12.3133,  -1.3133,  -0.3133, -15.3133],
         [ -3.4402,  -2.4402,  -1.4402,  -0.4402]]]
        tools.compare(res.numpy(), expect, delta=1e-3)


def test_logsoftmax1():
    """
    test logsoftmax axis=0
    Returns:
        None
    """
    data = np.array([[[-2.0, 3.0, -4.0, 5.0],
                      [3.0, -4.0, 5.0, -6.0],
                      [-7.0, -8.0, 8.0, 9.0]],
                     [[1.0, -2.0, -3.0, 4.0],
                      [-5.0, 6.0, 7.0, -8.0],
                      [6.0, 7.0, 8.0, 9.0]]]).astype('float32')
    my_log_softnmax = paddle.nn.LogSoftmax(0)
    with fluid.dygraph.guard():
        data = fluid.dygraph.to_variable(data)
        res = my_log_softnmax(data)
        expect = [[[-3.0486e+00, -6.7153e-03, -1.3133e+00, -3.1326e-01],
         [-3.3541e-04, -1.0000e+01, -2.1269e+00, -1.2693e-01],
         [-1.3000e+01, -1.5000e+01, -6.9315e-01, -6.9315e-01]],
        [[-4.8587e-02, -5.0067e+00, -3.1326e-01, -1.3133e+00],
         [-8.0003e+00, -4.5399e-05, -1.2693e-01, -2.1269e+00],
         [-2.2603e-06, -3.0590e-07, -6.9315e-01, -6.9315e-01]]]
        tools.compare(res.numpy(), expect, delta=1e-3)


def test_meshgrid():
    """
    test meshgrid
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([4, 5]).astype(np.float32)
        z = np.array([6, 7, 8]).astype(np.float32)
        paddle_x = to_variable(x)
        paddle_y = to_variable(y)
        paddle_z = to_variable(z)
        res = paddle.tensor.meshgrid([paddle_x, paddle_y, paddle_z])
        expect1 = [[[1., 1., 1.],
                 [1., 1., 1.]],
                [[2., 2., 2.],
                 [2., 2., 2.]],
                [[3., 3., 3.],
                 [3., 3., 3.]]]
        expect2 = [[[4., 4., 4.],
                     [5., 5., 5.]],
                    [[4., 4., 4.],
                     [5., 5., 5.]],
                    [[4., 4., 4.],
                     [5., 5., 5.]]]
        expect3 = [[[6., 7., 8.],
                     [6., 7., 8.]],
                    [[6., 7., 8.],
                     [6., 7., 8.]],
                    [[6., 7., 8.],
                     [6., 7., 8.]]]
        tools.compare(res[0].numpy(), expect1)
        tools.compare(res[1].numpy(), expect2)
        tools.compare(res[2].numpy(), expect3)


def test_arange():
    """
    test arange default
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = paddle.arange(1, 5, 0.5)
        expect = [1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 4.5000]
        tools.compare(x.numpy(), expect)


def test_arange1():
    """
    test arange dtype=int32
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = paddle.arange(1, 5, 1, dtype=np.int32)
        expect = [1, 2, 3, 4]
        tools.compare(x.numpy(), expect)


def test_arange2():
    """
    test arange dtype=int64
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = paddle.arange(1, 5, 1, dtype=np.int64)
        expect = [1, 2, 3, 4]
        tools.compare(x.numpy(), expect)


def test_arange3():
    """
    test arange dtype=float32
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = paddle.arange(1, 5, 0.5, dtype=np.float32)
        expect = [1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 4.5000]
        tools.compare(x.numpy(), expect)


def test_arange4():
    """
    test arange dtype=float64
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = paddle.arange(1, 5, 0.5, dtype=np.float64)
        expect = [1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 4.5000]
        tools.compare(x.numpy(), expect)


def test_arange5():
    """
    test arange name param BUG!!!!
        None
    """
    with fluid.dygraph.guard():
        x = paddle.arange(1, 5, 0.5, dtype=np.float64, name="ss")
        expect = [1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 4.5000]
        tools.compare(x.numpy(), expect)


def test_bmm():
    """
    test bmm
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        a = np.random.random(size=[3, 1, 2])
        b = np.random.random(size=[3, 2, 1])
        x = to_variable(a)
        y = to_variable(b)
        res = paddle.bmm(x, y)
        expect = [[[0.43382605]], [[0.40628374]], [[0.91274966]]]
        tools.compare(res.numpy(), expect, delta=1e-4)


def test_interpolate():
    """
    test interpolate
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        data = np.random.random((1, 1, 3, 3)).astype("float64")
        x = to_variable(data)
        align_corners = True
        out_shape = [4, 4]
        res = paddle.nn.functional.interpolate(x, size=out_shape, mode='BICUBIC', align_mode=0,
                                              align_corners=align_corners)
        expect = [[[[0.2485, 0.3909, 0.4601, 0.4109],
                  [0.2833, 0.6472, 0.6122, 0.2011],
                  [0.1859, 0.7745, 0.8299, 0.3159],
                  [0.0197, 0.6897, 0.9711, 0.6805]]]]
        tools.compare(res.numpy(), expect, delta=1e-4)


def test_interpolate1():
    """
    test interpolate1 align_corners=False
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        data = np.random.random((1, 1, 3, 3)).astype("float64")
        x = to_variable(data)
        align_corners = False
        out_shape = [4, 4]
        res = paddle.nn.functional.interpolate(x, size=out_shape, mode='BICUBIC', align_mode=0,
                                              align_corners=align_corners)
        expect = [[[[0.2353, 0.3570, 0.4414, 0.4277],
          [0.2518, 0.6105, 0.5774, 0.1763],
          [0.1195, 0.7352, 0.8122, 0.2951],
          [-0.0663, 0.6411, 0.9767, 0.6986]]]]
        tools.compare(res.numpy(), expect, delta=1e-4)


def test_NLLLoss():
    """
    test NLLLoss
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        input = np.random.random(size=(20, 200)).astype(np.float32)
        label = np.random.randint(0, 100, size=(20,)).astype(np.int64)
        p_input = to_variable(input)
        p_label = to_variable(label)
        nll_loss = paddle.nn.loss.NLLLoss()
        res = nll_loss(p_input, p_label)
        expect = [-0.5075191]
        tools.compare(res.numpy(), expect)


def test_NLLLoss1():
    """
    test NLLLoss add weight
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        input = np.random.random(size=(20, 200)).astype(np.float32)
        label = np.random.randint(0, 100, size=(20,)).astype(np.int64)
        weight = np.random.random(size=[200]).astype(np.float32)
        p_input = to_variable(input)
        p_label = to_variable(label)
        nll_loss = paddle.nn.loss.NLLLoss(to_variable(weight))
        res = nll_loss(p_input, p_label)
        expect = [-0.47225362]
        tools.compare(res.numpy(), expect)


def test_NLLLoss2():
    """
    test NLLLoss reducetion=sum
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        input = np.random.random(size=(20, 200)).astype(np.float32)
        label = np.random.randint(0, 100, size=(20,)).astype(np.int64)
        weight = np.random.random(size=[200]).astype(np.float32)
        p_input = to_variable(input)
        p_label = to_variable(label)
        nll_loss = paddle.nn.loss.NLLLoss(to_variable(weight), reduction="sum")
        res = nll_loss(p_input, p_label)
        expect = [-4.3605204]
        tools.compare(res.numpy(), expect)


def test_NLLLoss3():
    """
    test NLLLoss reducetion=None
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        input = np.random.random(size=(20, 200)).astype(np.float32)
        label = np.random.randint(0, 100, size=(20,)).astype(np.int64)
        weight = np.random.random(size=[200]).astype(np.float32)
        p_input = to_variable(input)
        p_label = to_variable(label)
        nll_loss = paddle.nn.loss.NLLLoss(to_variable(weight), reduction="none")
        res = nll_loss(p_input, p_label)
        expect = [-0.1155, -0.0369, -0.5154, -0.7624, -0.0933, -0.0631, -0.0307, -0.1075,
                -0.1835, -0.1925, -0.3282, -0.2857, -0.1193, -0.2945, -0.0721, -0.0174,
                -0.0599, -0.5841, -0.4217, -0.0769]
        tools.compare(res.numpy(), expect, delta=1e-4)


def test_std():
    """
    test std
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.random.uniform(1, 100, size=[3, 3])
        res = paddle.std(to_variable(x))
        expect = [31.62808741]
        tools.compare(res.numpy(), expect, delta=1e-4)


def test_std1():
    """
    test std keepdim=True
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.random.uniform(1, 100, size=[3, 3])
        res = paddle.std(to_variable(x), axis=[0, 1], keepdim=True)
        expect = [[31.62808741]]
        tools.compare(res.numpy(), expect, delta=1e-4)


def test_std2():
    """
    test std keepdim=True axis=1
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.random.uniform(1, 100, size=[3, 3])
        res = paddle.std(to_variable(x), axis=[1], keepdim=True)
        expect = [[10.57769871], [37.20946482], [47.52437458]]
        tools.compare(res.numpy(), expect, delta=1e-4)


def test_std3():
    """
    test std keepdim=True axis=1 unbiased=False
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.random.uniform(1, 100, size=[3, 3])
        res = paddle.std(to_variable(x), axis=[1], keepdim=True, unbiased=False)
        expect = [[ 8.63665483], [30.3814008], [38.80348936]]
        tools.compare(res.numpy(), expect, delta=1e-4)


def test_std4():
    """
    test std keepdim=True axis=1 unbiased=False name=ss
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.random.uniform(1, 100, size=[3, 3])
        aaa = to_variable(np.ones(shape=(3, 1)))
        res = paddle.std(to_variable(x), axis=[1], keepdim=True, unbiased=False, name="ss", out=aaa)
        expect = [[8.63665483], [30.3814008], [38.80348936]]
        tools.compare(aaa.numpy(), expect, delta=1e-4)


def test_clamp():
    """
    test clamp
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        x = to_variable(x)
        res = paddle.tensor.math.clamp(x, min=3, max=7)
        expect = [[3, 3, 3], [4, 5, 6], [7, 7, 7]]
        tools.compare(res.numpy(), expect)


def test_clamp1():
    """
    test clamp dtype=np.float64
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float64)
        x = to_variable(x)
        res = paddle.tensor.math.clamp(x, min=3, max=7)
        expect = [[3, 3, 3], [4, 5, 6], [7, 7, 7]]
        tools.compare(res.numpy(), expect)

def test_BCELoss():
    """
    test BCELoss
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        input = np.random.random(size=[100, 100, 33]).astype(np.float32)
        label = np.random.random(size=[100, 100, 33]).astype(np.float32)
        p_input = to_variable(input)
        p_label = to_variable(label)
        loss = paddle.nn.BCELoss()
        res = loss(p_input, p_label)
        expect = [1.0012524]
        tools.compare(res.numpy(), expect)


def test_BCELoss1():
    """
    test BCELoss weight param
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        input = np.random.random(size=[100, 100, 33]).astype(np.float32)
        label = np.random.random(size=[100, 100, 33]).astype(np.float32)
        weight = np.random.random(size=[100, 100, 33]).astype(np.float32)
        p_input = to_variable(input)
        p_label = to_variable(label)
        p_weight = to_variable(weight)
        loss = paddle.nn.BCELoss(weight=p_weight)
        res = loss(p_input, p_label)
        expect = [0.4997204]
        tools.compare(res.numpy(), expect)


def test_BCELoss2():
    """
    test BCELoss reduce=sum
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        input = np.random.random(size=[3, 33, 33]).astype(np.float32)
        label = np.random.randint(2, size=[3, 33, 33]).astype(np.float32)
        weight = np.random.random(size=[3, 33, 33]).astype(np.float32)
        p_input = to_variable(input)
        p_label = to_variable(label)
        p_weight = to_variable(weight)
        loss = paddle.nn.BCELoss(weight=p_weight, reduction="sum")
        res = loss(p_input, p_label)
        expect = [1641.069]
        tools.compare(res.numpy(), expect)


def test_BCELoss3():
    """
    test BCELoss reduce=none
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        input = np.random.random(size=[2, 2, 2]).astype(np.float32)
        label = np.random.randint(2, size=[2, 2, 2]).astype(np.float32)
        weight = np.random.random(size=[2, 2, 2]).astype(np.float32)
        p_input = to_variable(input)
        p_label = to_variable(label)
        p_weight = to_variable(weight)
        loss = paddle.nn.BCELoss(weight=p_weight, reduction="none")
        res = loss(p_input, p_label)
        expect = [[[0.1108, 0.2806],
                 [0.1455, 0.2964]],
                [[1.7994, 0.8336],
                 [0.0080, 0.0216]]]
        tools.compare(res.numpy(), expect, 1e-4)


def test_tril():
    """
    test tril
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        p_x = to_variable(x)
        res = paddle.tensor.tril(p_x)
        expect = [[1, 0, 0], [4, 5, 0], [7, 8, 9]]
        tools.compare(res.numpy(), expect)


def test_tril1():
    """
    test tril  diagonal=1
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        p_x = to_variable(x)
        res = paddle.tensor.tril(p_x, diagonal=1)
        expect = [[1, 2, 0], [4, 5, 6], [7, 8, 9]]
        tools.compare(res.numpy(), expect)


def test_tril2():
    """
    test tril  diagonal=-1
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        p_x = to_variable(x)
        res = paddle.tensor.tril(p_x, diagonal=-1)
        expect = [[0, 0, 0], [4, 0, 0], [7, 8, 0]]
        tools.compare(res.numpy(), expect)


def test_triu():
    """
    test triu
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        p_x = to_variable(x)
        res = paddle.tensor.triu(p_x)
        expect = [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
        tools.compare(res.numpy(), expect)


def test_triu1():
    """
    test triu diagonal=1
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        p_x = to_variable(x)
        res = paddle.tensor.triu(p_x, diagonal=1)
        expect = [[0, 2, 3], [0, 0, 6], [0, 0, 0]]
        tools.compare(res.numpy(), expect)


def test_triu2():
    """
    test triu diagonal=-1
    Returns:
        None
    """
    with fluid.dygraph.guard():
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        p_x = to_variable(x)
        res = paddle.tensor.triu(p_x, diagonal=-1)
        expect = [[1, 2, 3], [4, 5, 6], [0, 8, 9]]
        tools.compare(res.numpy(), expect)


def test_addmm():
    """
    test addmm
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = np.array([[3, 3], [3, 3]]).astype(np.float32)
        y = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
        p_i = to_variable(input)
        p_x = to_variable(x)
        p_y = to_variable(y)
        res = paddle.addmm(p_i, p_x, p_y, alpha=1, beta=1)
        expect = [[25., 31.],
                    [52., 67.]]
        tools.compare(res.numpy(), expect)


def test_addmm1():
    """
    test addmm broadcast
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = np.array([[3, 3]]).astype(np.float32)
        y = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
        p_i = to_variable(input)
        p_x = to_variable(x)
        p_y = to_variable(y)
        res = paddle.addmm(p_i, p_x, p_y, alpha=1, beta=1)
        expect = [[25., 31.],
                    [52., 67.]]
        tools.compare(res.numpy(), expect)


def test_addmm2():
    """
    test addmm broadcast beta=-1
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = np.array([[3, 3]]).astype(np.float32)
        y = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
        p_i = to_variable(input)
        p_x = to_variable(x)
        p_y = to_variable(y)
        res = paddle.addmm(p_i, p_x, p_y, alpha=1, beta=-1)
        expect = [[19., 25.],
                    [46., 61.]]
        tools.compare(res.numpy(), expect)


def test_addmm3():
    """
    test addmm broadcast alpha=2 beta=0
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = np.array([[3, 3]]).astype(np.float32)
        y = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
        x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
        p_i = to_variable(input)
        p_x = to_variable(x)
        p_y = to_variable(y)
        res = paddle.addmm(p_i, p_x, p_y, alpha=2, beta=0)
        expect = [[ 44.,  56.],
                [ 98., 128.]]
        tools.compare(res.numpy(), expect)


def test_index_sample():
    """
    test index_sample
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        index = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).astype(np.int32)
        x = to_variable(input)
        y = to_variable(index)
        res = paddle.index_sample(x, y)
        expect = [[1, 1, 1], [5, 5, 5], [9, 9, 9]]
        tools.compare(res.numpy(), expect)


def test_index_sample1():
    """
    test index_sample different shape
    Returns:
        None
    """
    with fluid.dygraph.guard():
        input = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        index = np.array([[0, ], [1, ], [2, ]]).astype(np.int32)
        x = to_variable(input)
        y = to_variable(index)
        res = paddle.index_sample(x, y)
        expect = [[1], [5], [9]]
        tools.compare(res.numpy(), expect)


def test_cholesky():
    """
    test cholesky
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.random.rand(3, 3)
        x_t = np.transpose(x, [1, 0])
        data = np.matmul(x, x_t)
        p = to_variable(data)
        res = paddle.cholesky(p, upper=False)
        expect = [[0.6581, 0.0000, 0.0000],
                [0.8090, 0.4530, 0.0000],
                [1.0841, 0.1849, 0.4033]]
        tools.compare(res.numpy(), expect, 1e-4)


def test_cholesky1():
    """
    test cholesky
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.random.rand(3, 3)
        x_t = np.transpose(x, [1, 0])
        data = np.matmul(x, x_t)
        p = to_variable(data)
        res = paddle.cholesky(p, upper=True)
        expect = [[0.6581, 0.8090, 1.0841],
            [0.0000, 0.4530, 0.1849],
            [0.0000, 0.0000, 0.4033]]
        tools.compare(res.numpy(), expect, 1e-4)


def test_CrossEntropyLoss():
    """
    test CrossEntropyLoss
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        input = np.random.random(size=[5, 100]).astype(np.float32)
        label = np.array([1, 2, 3, 4, 5]).astype(np.int64)
        weight = np.random.random(size=[5]).astype("float32")
        p_i = to_variable(input)
        p_l = to_variable(label)
        celoss = paddle.nn.loss.CrossEntropyLoss()
        res = celoss(p_i, p_l)
        expect = [4.575522]
        tools.compare(res.numpy(), expect)


def test_CrossEntropyLoss1():
    """
        test CrossEntropyLoss add weight
        Returns:
            None
        """
    with fluid.dygraph.guard():
        np.random.seed(33)
        input = np.random.random(size=[5, 100]).astype(np.float32)
        label = np.array([1, 2, 3, 4, 5]).astype(np.int64)
        weight = np.random.random(size=[100]).astype("float32")
        p_i = to_variable(input)
        p_l = to_variable(label)
        p_w = to_variable(weight)
        celoss = paddle.nn.loss.CrossEntropyLoss(weight=p_w)
        res = celoss(p_i, p_l)
        expect = [4.535555]
        tools.compare(res.numpy(), expect)


def test_CrossEntropyLoss2():
    """
        test CrossEntropyLoss reduction=sum
        Returns:
            None
        """
    with fluid.dygraph.guard():
        np.random.seed(33)
        input = np.random.random(size=[5, 100]).astype(np.float32)
        label = np.array([1, 2, 3, 4, 5]).astype(np.int64)
        weight = np.random.random(size=[100]).astype("float32")
        p_i = to_variable(input)
        p_l = to_variable(label)
        p_w = to_variable(weight)
        celoss = paddle.nn.loss.CrossEntropyLoss(reduction="sum")
        res = celoss(p_i, p_l)
        expect = [22.87761]
        tools.compare(res.numpy(), expect)


def test_CrossEntropyLoss3():
    """
        test CrossEntropyLoss reduction=none
        Returns:
            None
        """
    with fluid.dygraph.guard():
        np.random.seed(33)
        input = np.random.random(size=[5, 100]).astype(np.float32)
        label = np.array([1, 2, 3, 4, 5]).astype(np.int64)
        weight = np.random.random(size=[100]).astype("float32")
        p_i = to_variable(input)
        p_l = to_variable(label)
        p_w = to_variable(weight)
        celoss = paddle.nn.loss.CrossEntropyLoss(reduction="none")
        res = celoss(p_i, p_l)
        expect = [4.6951137, 4.709591, 4.709876, 4.5558195, 4.207209]
        tools.compare(res.numpy(), expect)


def test_log1p():
    """
    test log1p
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.random.random(size=(100, 100, 100))
        data = to_variable(x)
        res = paddle.log1p(data)
        expect = 0.38606708
        tools.compare(res.numpy().mean(), expect)


def test_rand():
    """
    test rand
    Returns:
        None
    """
    for i in range(3):
        with fluid.dygraph.guard():
            paddle.manual_seed(33)
            res = paddle.rand(shape=[3, 3])
            print(res.numpy())


def test_lstmcell():
    """
    test lstmcell
    Returns:
        None
    """
    input_size = 3
    hidden_size = 3
    batch_size = 2
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        fluid.default_main_program().random_seed = 33
        fluid.default_startup_program().random_seed = 33
        np.random.seed(33)
        param_attr = fluid.initializer.ConstantInitializer(value=1)
        bias_attr = fluid.initializer.ConstantInitializer(value=1)
        # named_cudnn_lstm = LSTMCell(hidden_size, input_size,
        #                             param_attr, bias_attr, use_cudnn_impl=False)
        cudnn_lstm = fluid.dygraph.rnn.LSTMCell(hidden_size, input_size, use_cudnn_impl=True)

        step_input_np = np.random.uniform(-0.1, 0.1, (
            batch_size, input_size)).astype('float64')
        pre_hidden_np = np.random.uniform(-0.1, 0.1, (
            batch_size, hidden_size)).astype('float64')
        pre_cell_np = np.random.uniform(-0.1, 0.1, (
            batch_size, hidden_size)).astype('float64')
        step_input_var = fluid.dygraph.to_variable(step_input_np)
        pre_hidden_var = fluid.dygraph.to_variable(pre_hidden_np)
        pre_cell_var = fluid.dygraph.to_variable(pre_cell_np)

        api_out = cudnn_lstm(step_input_var, pre_hidden_var, pre_cell_var)
        # named_api_out = named_cudnn_lstm(step_input_var, pre_hidden_var,
        #                                  pre_cell_var)
        # print(api_out[1].numpy())
        # print(api_out[0].numpy())
        expect_hidden = [[-5.24510955e-02, -2.60226713e-02, -1.77176250e-02],
                        [6.70252322e-05, 1.60950977e-02, 3.08377708e-02]]
        expect_cell = [[-0.10320148, -0.05285452, -0.03687387],
                        [0.00013388, 0.03373513, 0.06384785]]
        tools.compare(api_out[0].numpy(), expect_hidden)
        tools.compare(api_out[1].numpy(), expect_cell)


def test_lstmcell1():
    """
    test lstmcell use cudnn=false
    Returns:
        None
    """
    input_size = 3
    hidden_size = 3
    batch_size = 2
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        fluid.default_main_program().random_seed = 33
        fluid.default_startup_program().random_seed = 33
        np.random.seed(33)
        param_attr = fluid.initializer.ConstantInitializer(value=1)
        bias_attr = fluid.initializer.ConstantInitializer(value=1)
        # named_cudnn_lstm = LSTMCell(hidden_size, input_size,
        #                             param_attr, bias_attr, use_cudnn_impl=False)
        cudnn_lstm = fluid.dygraph.rnn.LSTMCell(hidden_size, input_size, use_cudnn_impl=False)

        step_input_np = np.random.uniform(-0.1, 0.1, (
            batch_size, input_size)).astype('float64')
        pre_hidden_np = np.random.uniform(-0.1, 0.1, (
            batch_size, hidden_size)).astype('float64')
        pre_cell_np = np.random.uniform(-0.1, 0.1, (
            batch_size, hidden_size)).astype('float64')
        step_input_var = fluid.dygraph.to_variable(step_input_np)
        pre_hidden_var = fluid.dygraph.to_variable(pre_hidden_np)
        pre_cell_var = fluid.dygraph.to_variable(pre_cell_np)

        api_out = cudnn_lstm(step_input_var, pre_hidden_var, pre_cell_var)
        # named_api_out = named_cudnn_lstm(step_input_var, pre_hidden_var,
        #                                  pre_cell_var)
        # print(api_out[1].numpy())
        # print(api_out[0].numpy())
        expect_hidden = [[-0.02546961, -0.02039802, -0.03682233],
                      [0.04039395, 0.03288361, -0.0227369]]
        expect_cell = [[-0.05112181, -0.04246271, -0.07330992],
                        [0.07909604, 0.0659461, -0.04493086]]
        tools.compare(api_out[0].numpy(), expect_hidden)
        tools.compare(api_out[1].numpy(), expect_cell)


def test_UpSample():
    """
    test UpSample
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.random.rand(1, 3, 6).astype("float64")
        data = to_variable(x)
        size = [3, ]
        align_corners = True
        upsample = paddle.nn.UpSample(
            size=size, mode="LINEAR", align_corners=align_corners, data_format="NCW")
        res = upsample(data)
        expect = [[[0.2485, 0.3356, 0.1850],
                 [0.0197, 0.5835, 0.3934],
                 [0.0796, 0.5734, 0.4941]]]
        tools.compare(res.numpy(), expect, delta=1e-3)


def test_UpSample1():
    """
    test UpSample test BILINEAR
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.random.rand(1, 3, 6, 9).astype("float64")
        data = to_variable(x)
        size = [3, 3]
        align_corners = True
        upsample = paddle.nn.UpSample(
            size=size, mode="BILINEAR", align_corners=align_corners, data_format="NCHW")
        res = upsample(data)
        expect = [[[[0.2485, 0.8704, 0.6805],
          [0.2283, 0.7222, 0.5163],
          [0.7316, 0.1366, 0.5001]],
         [[0.0595, 0.4279, 0.3441],
          [0.5376, 0.3061, 0.6620],
          [0.8507, 0.6089, 0.6727]],
         [[0.8077, 0.3185, 0.8259],
          [0.4832, 0.5181, 0.4941],
          [0.1329, 0.8070, 0.4061]]]]
        tools.compare(res.numpy(), expect, delta=1e-3)


def test_UpSample2():
    """
    test UpSample test BICUBIC
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.random.rand(1, 3, 6, 9).astype("float64")
        data = to_variable(x)
        size = [3, 3]
        align_corners = True
        upsample = paddle.nn.UpSample(
            size=size, mode="BICUBIC", align_corners=align_corners, data_format="NCHW")
        res = upsample(data)
        expect = [[[[0.2485, 0.8704, 0.6805],
          [0.1734, 0.7906, 0.5224],
          [0.7316, 0.1366, 0.5001]],

         [[0.0595, 0.4279, 0.3441],
          [0.5100, 0.2871, 0.6759],
          [0.8507, 0.6089, 0.6727]],

         [[0.8077, 0.3185, 0.8259],
          [0.4597, 0.5047, 0.4170],
          [0.1329, 0.8070, 0.4061]]]]
        tools.compare(res.numpy(), expect, delta=1e-3)


def test_UpSample3():
    """
    test UpSample test trilinear
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.random.rand(1, 3, 6, 9, 3).astype("float64")
        data = to_variable(x)
        size = [2, 2, 2]
        align_corners = True
        upsample = paddle.nn.UpSample(
            size=size, mode="TRILINEAR", align_corners=align_corners, data_format="NCDHW")
        res = upsample(data)
        expect = [[[[[0.2485, 0.4109],
           [0.2012, 0.9535]],
          [[0.2645, 0.8580],
           [0.1951, 0.4061]]],
         [[[0.8301, 0.5789],
           [0.6392, 0.2044]],
          [[0.4113, 0.1791],
           [0.6940, 0.1716]]],
         [[[0.1808, 0.1671],
           [0.1231, 0.9637]],
          [[0.2620, 0.7487],
           [0.4868, 0.9824]]]]]
        tools.compare(res.numpy(), expect, delta=1e-3)


def test_UpSample4():
    """
    test UpSample test trilinear align_corners=False mode=0
    Returns:
        None
    """
    with fluid.dygraph.guard():
        np.random.seed(33)
        x = np.random.rand(1, 3, 6, 9, 3).astype("float64")
        data = to_variable(x)
        size = [2, 2, 2]
        align_corners = False
        upsample = paddle.nn.UpSample(
            size=size, mode="TRILINEAR", align_corners=align_corners, data_format="NCDHW", align_mode=0)
        res = upsample(data)
        expect = [[[[[0.4861, 0.2611],
           [0.6224, 0.4009]],
          [[0.6927, 0.6781],
           [0.6420, 0.2470]]],
         [[[0.8460, 0.5089],
           [0.8478, 0.6063]],
          [[0.6867, 0.8512],
           [0.2138, 0.6325]]],
         [[[0.7176, 0.4754],
           [0.3148, 0.4544]],
          [[0.5984, 0.5547],
           [0.7635, 0.7526]]]]]
        tools.compare(res.numpy(), expect, delta=1e-3)
