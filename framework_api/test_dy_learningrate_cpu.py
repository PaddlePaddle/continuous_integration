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
"""test dygraph learningrate."""
import paddle.fluid as fluid
import numpy as np
import tools
import math

cpu = fluid.CPUPlace()

def test_PiecewiseDecay():
    """
    test PiecewiseDecay
    :return:
    """
    with fluid.dygraph.guard(cpu):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        classdim = 7
        x = fluid.dygraph.to_variable(np.arange(0, 21).astype('float32').reshape(3, 7))
        label = fluid.dygraph.to_variable(np.arange(0, 3).astype('int64').reshape(3, 1))
        # basic test
        boundaries = [10, 20]
        values = [1.0, 0.5, 0.1]
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0), parameter_list=fc.parameters())
        for step in range(30):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1 for i in range(10)] + [0.5 for i in range(10)] + [0.1 for i in range(10)]
        tools.compare(res, exp)

        # set step boundaries * 2
        boundaries = [20, 40]
        values = [1.0, 0.5, 0.1]
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0, step=2), parameter_list=fc.parameters())
        for step in range(30):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1 for i in range(10)] + [0.5 for i in range(10)] + [0.1 for i in range(10)]
        tools.compare(res, exp)

        # set begin=5 => 1*5 + 0.5 *10 + 0.1 * 15
        boundaries = [10, 20]
        values = [1.0, 0.5, 0.1]
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 5), parameter_list=fc.parameters())
        for step in range(30):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1 for i in range(5)] + [0.5 for i in range(10)] + [0.1 for i in range(15)]
        tools.compare(res, exp)


def test_CosineDecay():
    """
    test CosineDecay 余弦衰减学习率
    :return:
    """
    with fluid.dygraph.guard(cpu):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        classdim = 7
        # x = fluid.layers.data(name='x', shape=[3, 7], dtype='float32', append_batch_size=False)
        # label = fluid.layers.data(name='label', shape=[3, 1], dtype='int64', append_batch_size=False)
        x = fluid.dygraph.to_variable(np.arange(0, 21).astype('float32').reshape(3, 7))
        label = fluid.dygraph.to_variable(np.arange(0, 3).astype('int64').reshape(3, 1))
        label.stop_gradient = True
        # basic test
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.CosineDecay(
                learning_rate=1,
                step_each_epoch=3,
                epochs=3), parameter_list=fc.parameters())
        for epoch in range(3):
            for step in range(3):
                predict = fc(x)
                cost = fluid.layers.cross_entropy(input=predict, label=label)
                cost.backward()
                sgd_optimizer.minimize(cost)
                res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1, 1, 1, 0.75, 0.75, 0.75, 0.25, 0.25, 0.25]
        tools.compare(res, exp)

        # more epochs
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.CosineDecay(
                learning_rate=1,
                step_each_epoch=10,
                epochs=3), parameter_list=fc.parameters())
        for epoch in range(3):
            for step in range(10):
                predict = fc(x)
                cost = fluid.layers.cross_entropy(input=predict, label=label)
                cost.backward()
                sgd_optimizer.minimize(cost)
                res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
               0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        tools.compare(res, exp)

        # step = 2 allstep = 20
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.CosineDecay(
                learning_rate=1,
                step_each_epoch=20,
                step=2,
                epochs=3), parameter_list=fc.parameters())
        for epoch in range(3):
            for step in range(10):
                predict = fc(x)
                cost = fluid.layers.cross_entropy(input=predict, label=label)
                cost.backward()
                sgd_optimizer.minimize(cost)
                res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
               0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        tools.compare(res, exp)

        # step = 2 allstep = 20
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.CosineDecay(
                learning_rate=1,
                step_each_epoch=20,
                step=2,
                epochs=3), parameter_list=fc.parameters())
        for epoch in range(3):
            for step in range(10):
                predict = fc(x)
                cost = fluid.layers.cross_entropy(input=predict, label=label)
                cost.backward()
                sgd_optimizer.minimize(cost)
                res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
               0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        tools.compare(res, exp)

        # begin = 5 allstep = 15
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.CosineDecay(
                learning_rate=1,
                step_each_epoch=20,
                step=2,
                epochs=3), parameter_list=fc.parameters())
        for epoch in range(3):
            for step in range(10):
                predict = fc(x)
                cost = fluid.layers.cross_entropy(input=predict, label=label)
                cost.backward()
                sgd_optimizer.minimize(cost)
                res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
               0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        tools.compare(res, exp)


def test_ExponentialDecay():
    """
    test ExponentialDecay 指数衰减
    :return:
    """
    with fluid.dygraph.guard(cpu):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        classdim = 7
        x = fluid.dygraph.to_variable(np.arange(0, 21).astype('float32').reshape(3, 7))
        label = fluid.dygraph.to_variable(np.arange(0, 3).astype('int64').reshape(3, 1))
        label.stop_gradient = True
        # basic test
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.ExponentialDecay(
                learning_rate=1,
                decay_steps=3,
                decay_rate=0.5,
                staircase=False
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1.0, 0.7937005, 0.62996054, 0.5, 0.39685026, 0.31498027, 0.25, 0.19842514, 0.15749012]
        tools.compare(res, exp)

        # staircase = True
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.ExponentialDecay(
                learning_rate=1,
                decay_steps=3,
                decay_rate=0.5,
                staircase=True
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25]
        tools.compare(res, exp)

        # staircase = True begin = 1 相当于 全局step + begin
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.ExponentialDecay(
                learning_rate=1,
                decay_steps=3,
                decay_rate=0.5,
                begin=1,
                staircase=True
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.125]
        tools.compare(res, exp)

        # staircase = True step = 2 相当于 全局step*2 + begin
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.ExponentialDecay(
                learning_rate=1,
                decay_steps=3,
                decay_rate=0.5,
                step=2,
                staircase=True
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1.0, 1.0, 0.5, 0.25, 0.25, 0.125, 0.0625, 0.0625, 0.03125]
        tools.compare(res, exp)


def test_InverseTimeDecay():
    """
    test InverseTimeDecay  反时限衰减学习率
    :return:
    """
    with fluid.dygraph.guard(cpu):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        classdim = 7
        x = fluid.dygraph.to_variable(np.arange(0, 21).astype('float32').reshape(3, 7))
        label = fluid.dygraph.to_variable(np.arange(0, 3).astype('int64').reshape(3, 1))
        label.stop_gradient = True
        # basic test
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.InverseTimeDecay(
                learning_rate=1,
                decay_steps=3,
                decay_rate=0.5,
                staircase=False
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1.0, 0.85714287, 0.75, 0.6666667, 0.59999996, 0.54545456, 0.5, 0.4615385, 0.4285714]
        tools.compare(res, exp)

        # decay_rate = 1
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.InverseTimeDecay(
                learning_rate=1,
                decay_steps=3,
                decay_rate=1,
                staircase=False
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = []
        # 衰减算法
        for i in range(9):
            tmp = 1 / (1 + 1 * i / 3)
            exp.append(tmp)
        tools.compare(res, exp)

        # staircase = True
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.InverseTimeDecay(
                learning_rate=1,
                decay_steps=3,
                decay_rate=0.5,
                staircase=True
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1.0, 1.0, 1.0, 0.6666667, 0.6666667, 0.6666667, 0.5, 0.5, 0.5]
        tools.compare(res, exp)

        # begin = 1
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.InverseTimeDecay(
                learning_rate=1,
                decay_steps=3,
                decay_rate=0.5,
                begin=1,
                staircase=True
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1.0, 1.0, 0.6666667, 0.6666667, 0.6666667, 0.5, 0.5, 0.5, 0.4]
        tools.compare(res, exp)

        # step = 2
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.InverseTimeDecay(
                learning_rate=1,
                decay_steps=3,
                decay_rate=0.5,
                step=2,
                staircase=True
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        exp = [1.0, 1.0, 0.6666667, 0.5, 0.5, 0.4, 0.33333334, 0.33333334, 0.2857143]
        tools.compare(res, exp)


def test_NaturalExpDecay():
    """
    test NaturalExpDecay 自然指数衰减学习率
    :return:
    """
    with fluid.dygraph.guard(cpu):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        classdim = 7
        x = fluid.dygraph.to_variable(np.arange(0, 21).astype('float32').reshape(3, 7))
        label = fluid.dygraph.to_variable(np.arange(0, 3).astype('int64').reshape(3, 1))
        label.stop_gradient = True
        # basic test
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.NaturalExpDecay(
                learning_rate=1,
                decay_steps=3,
                decay_rate=0.5,
                staircase=False
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            tmp = 1 * math.exp(-0.5 * i / 3)
            exp.append(tmp)
        tools.compare(res, exp)

        # staircase = True
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.NaturalExpDecay(
                learning_rate=1,
                decay_steps=3,
                decay_rate=0.5,
                staircase=True
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            tmp = 1 * math.exp(-0.5 * math.floor(i / 3))
            exp.append(tmp)
        tools.compare(res, exp)

        # decay_rate = 0.3
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.NaturalExpDecay(
                learning_rate=1,
                decay_steps=3,
                decay_rate=0.3,
                staircase=True
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            tmp = 1 * math.exp(-0.3 * math.floor(i / 3))
            exp.append(tmp)
        tools.compare(res, exp)

        # decay_rate = 0.5 begin = 2
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.NaturalExpDecay(
                learning_rate=1,
                decay_steps=3,
                decay_rate=0.5,
                begin=2,
                staircase=True
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            tmp = 1 * math.exp(-0.5 * math.floor((i + 2) / 3))
            exp.append(tmp)
        tools.compare(res, exp)

        # decay_rate = 0.5 begin = 2 step = 2
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.NaturalExpDecay(
                learning_rate=1,
                decay_steps=3,
                decay_rate=0.5,
                begin=2,
                step=2,
                staircase=True
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            tmp = 1 * math.exp(-0.5 * math.floor((i * 2 + 2) / 3))
            exp.append(tmp)
        tools.compare(res, exp)


def test_PolynomialDecay():
    """
    test PolynomialDecay 多项式衰减学习率
    :return:
    """
    with fluid.dygraph.guard(cpu):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        classdim = 7
        x = fluid.dygraph.to_variable(np.arange(0, 21).astype('float32').reshape(3, 7))
        label = fluid.dygraph.to_variable(np.arange(0, 3).astype('int64').reshape(3, 1))
        label.stop_gradient = True
        # basic test
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.PolynomialDecay(
                learning_rate=1,
                decay_steps=5,
                end_learning_rate=0,
                power=1.0,
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            i = min(i, 5)
            tmp = (1 - 0) * (1 - i / 5) ** 1.0 + 0
            exp.append(tmp)
        tools.compare(res, exp)

        # power = 2.0
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.PolynomialDecay(
                learning_rate=1,
                decay_steps=5,
                end_learning_rate=0,
                power=2.0,
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            i = min(i, 5)
            tmp = (1 - 0) * (1 - i / 5) ** 2.0 + 0
            exp.append(tmp)
        tools.compare(res, exp)

        # decay_steps = 3.0
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.PolynomialDecay(
                learning_rate=1,
                decay_steps=3,
                end_learning_rate=0,
                power=1.0,
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            i = min(i, 3)
            tmp = (1 - 0) * (1 - i / 3) ** 1.0 + 0
            exp.append(tmp)
        tools.compare(res, exp)

        # end_learning_rate = 0.5
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.PolynomialDecay(
                learning_rate=1,
                decay_steps=7,
                end_learning_rate=0.5,
                power=1.0,
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            i = min(i, 7)
            tmp = (1 - 0.5) * (1 - i / 7) ** 1.0 + 0.5
            exp.append(tmp)
        tools.compare(res, exp)

        # learning_rate = 2
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.PolynomialDecay(
                learning_rate=2,
                decay_steps=7,
                end_learning_rate=0.5,
                power=1.0,
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            i = min(i, 7)
            tmp = (2 - 0.5) * (1 - i / 7) ** 1.0 + 0.5
            exp.append(tmp)
        tools.compare(res, exp)

        # learning_rate = 2 begin =1
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.PolynomialDecay(
                learning_rate=2,
                decay_steps=7,
                end_learning_rate=0.5,
                begin=1,
                power=1.0
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            i = min(i + 1, 7)
            tmp = (2 - 0.5) * (1 - i / 7) ** 1.0 + 0.5
            exp.append(tmp)
        tools.compare(res, exp)

        # learning_rate = 2 begin =1 step = 2
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.PolynomialDecay(
                learning_rate=2,
                decay_steps=7,
                end_learning_rate=0.5,
                begin=1,
                step=2,
                power=1.0
            ), parameter_list=fc.parameters())
        for step in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            i = min(i * 2 + 1, 7)
            tmp = (2 - 0.5) * (1 - i / 7) ** 1.0 + 0.5
            exp.append(tmp)
        tools.compare(res, exp)

        # cycle = True
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.PolynomialDecay(
                learning_rate=2,
                decay_steps=7,
                end_learning_rate=0.5,
                power=1.0,
                cycle=True
            ), parameter_list=fc.parameters())
        for step in range(20):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(20):
            lr = math.ceil(i / 7)
            if lr == 0:
                lr = 1
            tmp = 7 * lr
            tmp = (2 - 0.5) * (1 - i / tmp) ** 1.0 + 0.5
            exp.append(tmp)
        tools.compare(res, exp)

        # cycle = True begin = 2
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.PolynomialDecay(
                learning_rate=2,
                decay_steps=7,
                end_learning_rate=0.5,
                power=1.0,
                begin=2,
                cycle=True
            ), parameter_list=fc.parameters())
        for step in range(20):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(20):
            lr = math.ceil((i + 2) / 7)
            if lr == 0:
                lr = 1
            tmp = 7 * lr
            tmp = (2 - 0.5) * (1 - (i + 2) / tmp) ** 1.0 + 0.5
            exp.append(tmp)
        tools.compare(res, exp)

        # cycle = True begin = 2  step = 2
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.PolynomialDecay(
                learning_rate=2,
                decay_steps=7,
                end_learning_rate=0.5,
                power=1.0,
                begin=2,
                step=2,
                cycle=True
            ), parameter_list=fc.parameters())
        for step in range(20):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(20):
            lr = math.ceil((i * 2 + 2) / 7)
            if lr == 0:
                lr = 1
            tmp = 7 * lr
            tmp = (2 - 0.5) * (1 - (i * 2 + 2) / tmp) ** 1.0 + 0.5
            exp.append(tmp)
        tools.compare(res, exp)


def test_NoamDecay():
    """
    test NoamDecay Noam学习率衰减
    :return:
    """
    with fluid.dygraph.guard(cpu):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        classdim = 7
        x = fluid.dygraph.to_variable(np.arange(0, 21).astype('float32').reshape(3, 7))
        label = fluid.dygraph.to_variable(np.arange(0, 3).astype('int64').reshape(3, 1))
        label.stop_gradient = True
        # basic test
        d_model = 2
        warmup_steps = 2
        begin = 1
        step = 1
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.NoamDecay(
                d_model=d_model,
                warmup_steps=warmup_steps,
                begin=begin,
                step=step
            ), parameter_list=fc.parameters())
        for i in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            lr = np.power(d_model, -0.5) * np.min([np.power((i * step + begin), -0.5),
                        np.power(warmup_steps, -1.5) * (i * step + begin)])
            exp.append(lr)
        tools.compare(res, exp)

        # d_model = 5
        d_model = 5
        warmup_steps = 2
        begin = 1
        step = 1
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.NoamDecay(
                d_model=d_model,
                warmup_steps=warmup_steps,
                begin=begin,
                step=step
            ), parameter_list=fc.parameters())
        for i in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            lr = np.power(d_model, -0.5) * np.min([np.power((i * step + begin), -0.5),
                                                   np.power(warmup_steps, -1.5) * (i * step + begin)])
            exp.append(lr)
        tools.compare(res, exp)

        # d_model = 5 warmup_steps = 1
        d_model = 5
        warmup_steps = 1
        begin = 1
        step = 1
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.NoamDecay(
                d_model=d_model,
                warmup_steps=warmup_steps,
                begin=begin,
                step=step
            ), parameter_list=fc.parameters())
        for i in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            lr = np.power(d_model, -0.5) * np.min([np.power((i * step + begin), -0.5),
                                                   np.power(warmup_steps, -1.5) * (i * step + begin)])
            exp.append(lr)
        tools.compare(res, exp)

        # d_model = 5 warmup_steps = 5
        d_model = 5
        warmup_steps = 5
        begin = 3
        step = 2
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.NoamDecay(
                d_model=d_model,
                warmup_steps=warmup_steps,
                begin=begin,
                step=step
            ), parameter_list=fc.parameters())
        for i in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            lr = np.power(d_model, -0.5) * np.min([np.power((i * step + begin), -0.5),
                                                   np.power(warmup_steps, -1.5) * (i * step + begin)])
            exp.append(lr)
        tools.compare(res, exp)

        # d_model = 5 warmup_steps = 5 begin = 3 step = 2
        d_model = 5
        warmup_steps = 5
        begin = 1
        step = 1
        res = []
        fc = fluid.dygraph.Linear(input_dim=7, output_dim=classdim, act='softmax')
        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.dygraph.NoamDecay(
                d_model=d_model,
                warmup_steps=warmup_steps,
                begin=begin,
                step=step
            ), parameter_list=fc.parameters())
        for i in range(9):
            predict = fc(x)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            cost.backward()
            sgd_optimizer.minimize(cost)
            res.append(sgd_optimizer._global_learning_rate().numpy()[0])
        # 衰减算法
        exp = []
        for i in range(9):
            lr = np.power(d_model, -0.5) * np.min([np.power((i * step + begin), -0.5),
                                                   np.power(warmup_steps, -1.5) * (i * step + begin)])
            exp.append(lr)
        tools.compare(res, exp)



