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
"""test static trigonometric."""
import paddle.fluid as fluid
import numpy as np
import math
import tools


def test_abs():
    """
    test abs
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.abs(data)
            x = np.array([1, -1]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=[result.name])[0][0]
                tools.compare(res, 1.0)


def test_abs1():
    """
    test abs with name=abs
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.abs(data, name="abs")
            x = np.array([1, -1]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["abs.tmp_0"])[0][0]
                tools.compare(res, 1.0)


def test_acos():
    """
    test acos
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.acos(data, name="abs")
            x = np.array([-0.8183, 0.4912, -0.6444, 0.0371]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=[result.name])[0]
            tools.compare(res, [2.5292435, 1.0573294, 2.2710347, 1.5336878],
                          1e-5)


def test_acos1():
    """
    test acos with name = acos
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.acos(data, name="acos")
            x = np.array([-0.8183, 0.4912, -0.6444, 0.0371]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["acos.tmp_0"])[0]
            tools.compare(res, [2.5292435, 1.0573294, 2.2710347, 1.5336878],
                          1e-5)


def test_asin():
    """
    test asin
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.asin(data)
            x = np.array([-0.8183, 0.4912, -0.6444, 0.0371]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=[result.name])[0]
            tools.compare(
                res, [-0.95844716, 0.51346684, -0.7002384, 0.03710851], 1e-5)


def test_asin1():
    """
    test asin with name=asin
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.asin(data, name="asin")
            x = np.array([-0.8183, 0.4912, -0.6444, 0.0371]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["asin.tmp_0"])[0]
            tools.compare(
                res, [-0.95844716, 0.51346684, -0.7002384, 0.03710851], 1e-5)


def test_atan():
    """
    test atan
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.atan(data)
            x = np.array([-0.8183, 0.4912, -0.6444, 0.0371]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=[result.name])[0]
            tools.compare(
                res, [-0.6858003, 0.45658287, -0.5724284, 0.03708299], 1e-5)


def test_atan1():
    """
    test atan with name=atan
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.atan(data, name="atan")
            x = np.array([-0.8183, 0.4912, -0.6444, 0.0371]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["atan.tmp_0"])[0]
            tools.compare(
                res, [-0.6858003, 0.45658287, -0.5724284, 0.03708299], 1e-5)


def test_ceil():
    """
    test ceil
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.ceil(data, name="ceil")
            x = np.array([-0.8183, 0.4912, -0.6444, 0.0371]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["ceil.tmp_0"])[0]
            tools.compare(res, [0, 1, 0, 1], 1e-6)


def test_cos():
    """
    test cos
    input is 0, pi, 2 * pi / 3, pi / 2
    expect is 1, -1, -0.5, 0
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.cos(data, name="cos")
            x = np.array(
                [0, math.pi, 2 * math.pi / 3, math.pi / 2]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["cos.tmp_0"])[0]
            tools.compare(res, [1, -1, -0.5, 0], 1e-6)


def test_exp():
    """
    test cos
    input is 0, 1, 3, -3
    expect is 1, e, e3, 1/e3
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.exp(data, name="exp")
            x = np.array([0, 1, 3, -3]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["exp.tmp_0"])[0]
            tools.compare(res, [1, math.e, math.e**3, math.e**-3], 1e-6)


def test_floor():
    """
    test floor
    input is -0.8183, 0.4912, -0.6444, 0.0371
    expect is -1, 0, -1, 0
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.floor(data, name="floor")
            x = np.array([-0.8183, 0.4912, -0.6444, 0.0371]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["floor.tmp_0"])[0]
            tools.compare(res, [-1, 0, -1, 0], 1e-6)


def test_hard_shrink():
    """
    test hard_shrink
    | out = x,if x>λ
    | out = x,if x<−λ
    | out = 0,otherwise
    input is -1, -2, 0, 0.3, 2.7, 1.3
    expect is 0, -2, 0, 0, 2.7, 1.3
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.hard_shrink(data, threshold=1)
            x = np.array([-1, -2, 0, 0.3, 2.7, 1.3]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["hard_shrink_0.tmp_0"])[0]
            tools.compare(res, [0, -2, 0, 0, 2.7, 1.3], 1e-6)


def test_logsigmoid():
    """
    test logsigmoid

    input is -1, -2, 0, 2.7
    expect is log(sigmoid(-1, -2, 0, 2.7))
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.logsigmoid(data, name="logsigmoid")
            x = np.array([-1, -2, 0, 2.7]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["logsigmoid.tmp_0"])[0]
            tools.compare(res, [
                math.log(tools.sigmoid(-1)),
                math.log(tools.sigmoid(-2)),
                math.log(tools.sigmoid(0)),
                math.log(tools.sigmoid(2.7)),
            ], 1e-6)


def test_reciprocal():
    """
    test reciprocal
    input is -2, -1, 0, 1, 5
    expect is -0.5, -1, inf, 1, 0.2
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.reciprocal(data, name="reciprocal")
            x = np.array([-2, -1, 0, 1, 5]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["reciprocal.tmp_0"])[0]
            tools.compare(res, [-0.5, -1, float("inf"), 1, 0.2], 1e-6)


def test_round():
    """
    test round
    input is -1.5, -0.32, 0, 0.55, 1.3
    expect is -2, 0, 0, 1, 1
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.round(data, name="round")
            x = np.array([-1.5, -0.32, 0, 0.55, 1.3]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["round.tmp_0"])[0]
            tools.compare(res, [-2, 0, 0, 1, 1], 1e-6)


def test_rsqrt():
    """
    test rsqrt
    input is -1, 0, 1, 4, 16
    expect is inf, inf, 1, 0.5, 0.25
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.rsqrt(data, name="rsqrt")
            x = np.array([0, 1, 4, 16]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["rsqrt.tmp_0"])[0]
            tools.compare(res, [float("inf"), 1, 0.5, 0.25], 1e-6)


def test_sigmoid():
    """
    test sigmoid
    input is -1, -2, 0, 2.7
    expect is sigmoid(-1, -2, 0, 2.7)
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.sigmoid(data, name="sigmoid")
            x = np.array([-1, -2, 0, 2.7]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["sigmoid.tmp_0"])[0]
            tools.compare(res, [
                tools.sigmoid(-1), tools.sigmoid(-2), tools.sigmoid(0),
                tools.sigmoid(2.7)
            ], 1e-6)


def test_sin():
    """
    test sin
    input is pi/6, pi/3, pi/2, 3pi/4, 3pi/2
    expect is 0.5, 0.8660254, 1, 0.70710678, -1
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.sin(data, name="sin")
            x = np.array([
                math.pi / 6, math.pi / 3, math.pi / 2, 3 * math.pi / 4,
                3 * math.pi / 2
            ]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["sin.tmp_0"])[0]
            tools.compare(res, [0.5, 0.8660254, 1, 0.70710678, -1], 1e-6)


def test_softplus():
    """
    test sin
    input is -2, -1, 0, 1, 5
    expect is ln(1 + ex)
    :return:
    """

    def softplus(x):
        """
        ln(1 + ex)
        :param x:
        :return:
        """
        return math.log(1 + math.e**x, math.e)

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.softplus(data, name="softplus")
            x = np.array([-2, -1, 0, 1, 5]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["softplus.tmp_0"])[0]
            tools.compare(res, [
                softplus(-2), softplus(-1), softplus(0), softplus(1),
                softplus(5)
            ], 1e-6)


def test_softshrink():
    """
    test softshrink
    | out = x−λ,if x>λ
    | out = x+λ,if x<−λ
    | out = 0,otherwise
    input is -2, -1, 0, 1, 5
    lambda us 0.5
    expect is -1.5, -0.5, 0.5, 0.5, 4.5
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.softshrink(data, alpha=0.5)
            x = np.array([-2, -1, 0, 1, 5]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["softshrink_0.tmp_0"])[0]
            tools.compare(res, [-1.5, -0.5, 0, 0.5, 4.5], 1e-6)


def test_softsign():
    """
    test softsign
    | out = x / (1 + |x|)
    input is -2, -1, 0, 1, 5
    expect is -0.66666667, -0.5, 0, 0.5, 0.83333333
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.softsign(data, "softsign")
            x = np.array([-2, -1, 0, 1, 5]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["softsign.tmp_0"])[0]
            tools.compare(res, [-0.66666667, -0.5, 0, 0.5, 0.83333333], 1e-6)


def test_sqrt():
    """
    test sqrt
    | out = sqrt(x)
    input is 0, 1, 4, 9, 16
    expect is sqrt(x)
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.sqrt(data, "sqrt")
            x = np.array([0, 1, 4, 9, 16]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["sqrt.tmp_0"])[0]
            tools.compare(res, [0, 1, 2, 3, 4], 1e-6)


def test_square():
    """
    test square
    | out = square(x)
    input is -2, -1, 0, 1, 2
    expect is 4, 1, 0, 1, 4
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.square(data, "square")
            x = np.array([-2, -1, 0, 1, 2]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["square.tmp_0"])[0]
            tools.compare(res, [4, 1, 0, 1, 4], 1e-6)


def test_tanh():
    """
    test tanh
    | out = tanh(x)
    input is -2, -1, 0, 1, 2
    expect is tanh(x)
    :return:
    """

    def tanh(x):
        """
        ex - e-x / ex + e-x
        :param x:
        :return:
        """
        return (math.e**x - math.e**-x) / (math.e**x + math.e**-x)

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.tanh(data, "tanh")
            x = np.array([-2, -1, 0, 1, 2]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["tanh.tmp_0"])[0]
            tools.compare(res, [tanh(-2), tanh(-1), tanh(0), tanh(1), tanh(2)],
                          1e-6)


def test_tanh_shrink():
    """
    test tanh_shrink
    | out = tanh(x)
    input is -2, -1, 0, 1, 2
    expect is tanh(x)
    :return:
    """

    def tanh_shrink(x):
        """
        x - (ex - e-x / ex + e-x)
        :param x:
        :return:
        """
        return x - (math.e**x - math.e**-x) / (math.e**x + math.e**-x)

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.tanh_shrink(data, "tanh_shrink")
            x = np.array([-2, -1, 0, 1, 2]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["tanh_shrink.tmp_0"])[0]
            tools.compare(res, [
                tanh_shrink(-2), tanh_shrink(-1), tanh_shrink(0),
                tanh_shrink(1), tanh_shrink(2)
            ], 1e-6)


def test_thresholded_relu():
    """
    test thresholded_relu
    | out = thresholded_relu(x)
    threshold = 1.5
    input is -2, -1, 0, 1, 2
    expect is 0, 0, 0, 0, 2
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[1])
            result = fluid.layers.thresholded_relu(data, 1.5)
            x = np.array([-2, -1, 0, 1, 2]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            for i in range(len(x)):
                res = exe.run(compiled_prog,
                              feed={"input": x},
                              fetch_list=["thresholded_relu_0.tmp_0"])[0]
            tools.compare(res, [0, 0, 0, 0, 2], 1e-6)


def test_uniform_random():
    """
    test uniform_random
    default
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            result = fluid.layers.uniform_random(shape=[3, 3], seed=33)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            res = exe.run(compiled_prog, fetch_list=[result])[0]
            expect = [[-0.99851644, 0.6121416, 0.6892719],
                      [-0.15365571, 0.8850763, -0.48047507],
                      [0.9884044, -0.7314464, 0.35180688]]
            tools.compare(res, expect, 1e-6)


def test_uniform_random1():
    """
    test uniform_random
    shape = tuple(3, 3)
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            result = fluid.layers.uniform_random(shape=(3, 3), seed=33)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            res = exe.run(compiled_prog, fetch_list=[result])[0]
            expect = [[-0.99851644, 0.6121416, 0.6892719],
                      [-0.15365571, 0.8850763, -0.48047507],
                      [0.9884044, -0.7314464, 0.35180688]]
            tools.compare(res, expect, 1e-6)


def test_uniform_random2():
    """
    test uniform_random
    shape = list (3, 2) max = 2 min = -2
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            result = fluid.layers.uniform_random(
                shape=(3, 2), max=2, min=-2, seed=33)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            res = exe.run(compiled_prog, fetch_list=[result])[0]
            print(res)
            expect = [[-1.9970329, 1.2242832], [1.3785439, -0.30731142],
                      [1.7701526, -0.96095014]]
            tools.compare(res, expect, 1e-6)


def test_cumsum():
    """
    test cumsum
    threshold = 1.5
    input is [[1, 1], [2, 3], [4, 5]]
    axis = 0
    expect is [[1, 1], [3, 4], [7, 9]]
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[3, 2])
            result = fluid.layers.cumsum(data, axis=0)
            x = np.array([[1, 1], [2, 3], [4, 5]]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            res = exe.run(compiled_prog, feed={"input": x},
                          fetch_list=[result])[0]
            expect = [[1, 1], [3, 4], [7, 9]]
            tools.compare(res, expect, 1e-6)


def test_cumsum1():
    """
    test cumsum
    threshold = 1.5
    input is [[1, 1], [2, 3], [4, 5]]
    axis = 1
    expect is [[1, 2], [2, 5], [4, 9]]
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[3, 2])
            result = fluid.layers.cumsum(data, axis=1)
            x = np.array([[1, 1], [2, 3], [4, 5]]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            res = exe.run(compiled_prog, feed={"input": x},
                          fetch_list=[result])[0]
            expect = [[1, 2], [2, 5], [4, 9]]
            tools.compare(res, expect, 1e-6)


def test_cumsum2():
    """
    test cumsum
    threshold = 1.5
    input is [[1, 1], [2, 3], [4, 5]]
    axis = default
    expect is [[1, 2], [2, 5], [4, 9]]
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[3, 2])
            result = fluid.layers.cumsum(data)
            x = np.array([[1, 1], [2, 3], [4, 5]]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            res = exe.run(compiled_prog, feed={"input": x},
                          fetch_list=[result])[0]
            expect = [[1, 2], [2, 5], [4, 9]]
            tools.compare(res, expect, 1e-6)


def test_cumsum3():
    """
    test cumsum
    threshold = 1.5
    input is [[1, 1], [2, 3], [4, 5]]
    axis = default
    expect is [[2, 1], [5, 3], [9, 5]]
    reverse = True
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[3, 2])
            result = fluid.layers.cumsum(data, reverse=True)
            x = np.array([[1, 1], [2, 3], [4, 5]]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            res = exe.run(compiled_prog, feed={"input": x},
                          fetch_list=[result])[0]
            expect = [[2, 1], [5, 3], [9, 5]]
            tools.compare(res, expect, 1e-6)


def test_cumsum4():
    """
    test cumsum
    threshold = 1.5
    input is [[1, 1], [2, 3], [4, 5]]
    axis = 0
    expect is [[0, 0], [1, 1], [3, 4]]
    exclusive = True
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[3, 2])
            result = fluid.layers.cumsum(data, axis=0, exclusive=True)
            x = np.array([[1, 1], [2, 3], [4, 5]]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            res = exe.run(compiled_prog, feed={"input": x},
                          fetch_list=[result])[0]
            expect = [[0, 0], [1, 1], [3, 4]]
            tools.compare(res, expect, 1e-6)


def test_cumsum5():
    """
    test cumsum
    threshold = 1.5
    input is [[1, 1], [2, 3], [4, 5]]
    axis = default
    expect is [[1, 0], [3, 0], [5, 0]]
    exclusive = True
    reverse = True
    :return:
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = fluid.layers.data(name="input", shape=[3, 2])
            result = fluid.layers.cumsum(data, reverse=True, exclusive=True)
            x = np.array([[1, 1], [2, 3], [4, 5]]).astype(np.float32)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            compiled_prog = fluid.compiler.CompiledProgram(main_program)
            res = exe.run(compiled_prog, feed={"input": x},
                          fetch_list=[result])[0]
            expect = [[1, 0], [3, 0], [5, 0]]
            tools.compare(res, expect, 1e-6)
