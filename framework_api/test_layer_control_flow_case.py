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
"""test layer control flow."""
import paddle.fluid as fluid
import numpy as np
from functools import partial
import tools


def test_case_base():
    """
    control flow case的基本功能
    """

    def fn_1():
        return fluid.layers.fill_constant(
            shape=[4, 2], dtype='float32', value=1.3)

    def fn_2():
        return fluid.layers.fill_constant(shape=[4, 2], dtype='int32', value=2)

    def fn_3():
        return fluid.layers.fill_constant(shape=[4, 3], dtype='int32', value=3)

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        x = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.3)
        y = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.1)
        z = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.2)
        pred_1 = fluid.layers.less_than(y, x)  # true: 0.1 < 0.3
        pred_2 = fluid.layers.equal(z, x)  # false: 0.2 < 0.3
        #调用fn1
        out_0 = fluid.layers.case(
            pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3)
        #调用fn2
        out_1 = fluid.layers.case(
            pred_fn_pairs=[(pred_2, fn_2), (pred_1, fn_2)], default=fn_3)
        #调用default fn3
        out_2 = fluid.layers.case(
            pred_fn_pairs=[(pred_2, fn_1), (pred_2, fn_2)], default=fn_3)
        #no default fn2
        out_3 = fluid.layers.case(
            pred_fn_pairs=[(pred_2, fn_1), (pred_1, fn_2)])
        #no default fn2, and fn2 is false
        out_4 = fluid.layers.case(
            pred_fn_pairs=[(pred_2, fn_2), (pred_2, fn_1)])
        #use ==
        out_5 = fluid.layers.case(pred_fn_pairs=[(x == y, fn_1), (x > y, fn_2)])

        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(main_program,
                      fetch_list=[out_0, out_1, out_2, out_3, out_4, out_5])
        assert np.allclose(res[0], 1.3)
        assert np.allclose(res[1], 2)
        assert np.allclose(res[2], 3)
        assert np.allclose(res[3], 2)
        assert np.allclose(res[4], 1.3)
        assert np.allclose(res[5], 2)
        assert np.allclose(res[4], 1.3)


def test_case_nested():
    """
    case语句的嵌套功能
    """

    def fn_1(x=1):
        var_1 = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=1.1)
        var_2 = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=2.2)
        out = fluid.layers.case(pred_fn_pairs=[(var_1 < var_2, partial(
            fluid.layers.fill_constant, shape=[1], dtype='int32', value=x)),
                                               (var_1 == var_2, partial(
                                                   fluid.layers.fill_constant,
                                                   shape=[2],
                                                   dtype='int32',
                                                   value=x))])
        return out

    def fn_2(x=2):
        var_1 = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=1.1)
        var_2 = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=2.2)
        out = fluid.layers.case(pred_fn_pairs=[(var_1 < var_2, partial(
            fn_1, x=x)), (var_1 == var_2, partial(
                fluid.layers.fill_constant, shape=[2], dtype='int32',
                value=x))])
        return out

    def fn_3():
        var_1 = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=1.1)
        var_2 = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=2.2)
        out = fluid.layers.case(pred_fn_pairs=[(var_1 < var_2, partial(
            fn_2, x=3)), (var_1 == var_2, partial(
                fluid.layers.fill_constant, shape=[2], dtype='int32',
                value=3))])
        return out

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        x = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.3)
        y = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.1)
        z = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.2)
        pred_1 = fluid.layers.less_than(y, x)  # True: 0.1 < 0.3
        pred_2 = fluid.layers.less_than(x, z)  # false: 0.3 < 0.2
        #fn_1
        out_1 = fluid.layers.case(
            pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3)
        #fn_2
        out_2 = fluid.layers.case(
            pred_fn_pairs=[(pred_2, fn_1), (pred_1, fn_2)], default=fn_3)
        #fn_3
        out_3 = fluid.layers.case(
            pred_fn_pairs=[(pred_2, fn_1), (pred_2, fn_2)], default=fn_3)

        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(main_program, fetch_list=[out_1, out_2, out_3])
        assert np.allclose(res[0], 1)
        assert np.allclose(res[1], 2)
        assert np.allclose(res[2], 3)


def test_case_with_cond():
    """
    control flow case with cond
    """

    def fn_1(x=1):
        var_1 = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=1.1)
        var_2 = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=2.2)
        out = fluid.layers.case(pred_fn_pairs=[(var_1 < var_2, partial(
            fluid.layers.fill_constant, shape=[1], dtype='int32', value=x)),
                                               (var_1 == var_2, partial(
                                                   fluid.layers.fill_constant,
                                                   shape=[2],
                                                   dtype='int32',
                                                   value=x))])
        return out

    def fn_2(x=2):
        var_1 = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=1.1)
        var_2 = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=2.2)
        out = fluid.layers.case(pred_fn_pairs=[(var_1 < var_2, partial(
            fn_1, x=x)), (var_1 == var_2, partial(
                fluid.layers.fill_constant, shape=[2], dtype='int32',
                value=x))])
        return out

    def fn_3():
        var_1 = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=1.1)
        var_2 = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=2.2)
        out = fluid.layers.case(pred_fn_pairs=[(var_1 < var_2, partial(
            fn_2, x=3)), (var_1 == var_2, partial(
                fluid.layers.fill_constant, shape=[2], dtype='int32',
                value=3))])
        return out

    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        x = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.3)
        y = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.1)
        z = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.2)
        pred_1 = fluid.layers.less_than(y, x)  # True: 0.1 < 0.3
        pred_2 = fluid.layers.less_than(x, z)  # false: 0.3 < 0.2
        #fn_1
        out_1 = fluid.layers.case(
            pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3)
        #fn_2
        out_2 = fluid.layers.case(
            pred_fn_pairs=[(pred_2, fn_1), (pred_1, fn_2)], default=fn_3)
        #fn_3
        out_3 = fluid.layers.case(
            pred_fn_pairs=[(pred_2, fn_1), (pred_2, fn_2)], default=fn_3)
        a = fluid.layers.fill_constant(shape=[1], dtype='float32', value=1.23)
        b = fluid.layers.fill_constant(shape=[1], dtype='float32', value=1.24)
        out = fluid.layers.cond(fluid.layers.less_than(a, b),
                                lambda: fluid.layers.cond(a - b < -1,
                                                lambda: fluid.layers.elementwise_add(out_2, out_3),
                                                lambda: fluid.layers.elementwise_mul(out_2, out_3)),
                                lambda: fluid.layers.cond(fluid.layers.equal(a, b),
                                                lambda: fluid.layers.elementwise_sub(out_1, out_2),
                                                lambda: fluid.layers.elementwise_pow(out_1, out_2))
                                )
        out.stop_gradient = False
        fluid.backward.append_backward(out)
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(main_program, fetch_list=[out])
        assert np.allclose(res, [6])


def test_case_linear():
    """
    简单的线性模型中使用 case
    """
    main_program = fluid.Program()
    startup_program = fluid.Program()
    startup_program.random_seed = 1
    main_program.random_seed = 1
    with fluid.program_guard(main_program, startup_program):
        outputs = np.asarray(
            [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16)])
        print(outputs)
        res = []
        for i in range(4):
            # 假设方程式为 y=4a+6b+7c+2d
            y = 4 * outputs[i][0] + 6 * outputs[i][1] + 7 * outputs[i][
                2] + 2 * outputs[i][3]
            res.append([y])
        # 定义数据
        train_data = np.array(outputs).astype('float32')
        y_true = np.array(res).astype('float32')
        #定义网络
        x = fluid.layers.data(name="x", shape=[4], dtype='float32')
        y = fluid.layers.data(name="y", shape=[1], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        #定义损失函数
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)

        #定义优化方法
        def sgd_optimizer(learning_rate):
            optimizer = fluid.optimizer.SGDOptimizer(
                learning_rate=learning_rate)
            optimizer.minimize(avg_cost)

        x = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.3)
        y = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.1)
        z = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.2)
        pred_1 = fluid.layers.less_than(y, x)  # True: 0.1 < 0.3
        pred_2 = fluid.layers.less_than(x, z)  # false: 0.3 < 0.2
        out = fluid.layers.case(
            pred_fn_pairs=[(pred_2, partial(
                sgd_optimizer, learning_rate=0.001)), (pred_2, partial(
                    sgd_optimizer, learning_rate=0.002))],
            default=partial(
                sgd_optimizer, learning_rate=0.005))
        #参数初始化
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        actual_cost = []
        for i in range(4):
            outs = exe.run(
                feed={'x': train_data[i:i + 1],
                      'y': y_true[i:i + 1]},
                fetch_list=[
                    x, fluid.layers.reduce_sum(x), y_predict.name, avg_cost.name
                ])
            print(outs)
            actual_cost.append(outs[3])
        print("asdasd")
        print(actual_cost)
        expect = [[1919.809], [8538.607], [10656.933], [247723.22]]
        tools.compare(actual_cost, expect)
