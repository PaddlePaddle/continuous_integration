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
"""test dygraph layer."""
import paddle
from paddle.fluid.dygraph.base import to_variable
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
import numpy as np
import tools
import math
import platform

place = fluid.CPUPlace()


def test_Linear():
    """
    test linear

    Returns: None

    """
    with fluid.dygraph.guard(place):
        data = np.ones(shape=(2, 3, 3), dtype=np.float32).reshape(2, 9)
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        # wx+b w=1 b=1 固定值初始化
        fc = Linear(
            9,
            3,
            param_attr=fluid.initializer.ConstantInitializer(value=1.0),
            bias_attr=fluid.initializer.ConstantInitializer(value=1.0))
        data = to_variable(data)
        conv = fc(data)
        exp1 = np.array([[10., 10., 10.], [10., 10., 10.]])
        tools.compare(conv.numpy(), exp1)
        # 随机高斯分布初始化
        fc = Linear(
            9, 3, param_attr=fluid.initializer.NormalInitializer(seed=seed))
        conv = fc(data)
        exp2 = conv.numpy()
        fc = Linear(
            9, 3, param_attr=fluid.initializer.NormalInitializer(seed=seed))
        conv = fc(data)
        tools.compare(conv.numpy(), exp2)
        # 激活函数
        fc = Linear(
            9,
            3,
            param_attr=fluid.initializer.ConstantInitializer(value=-1),
            act="relu")
        conv = fc(data)
        exp3 = np.zeros(shape=(2, 3), dtype=np.float32)
        tools.compare(conv.numpy(), exp3)
        fc = Linear(
            9,
            3,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            act="relu")
        conv = fc(data)
        exp4 = np.array([[9., 9., 9.], [9., 9., 9.]])
        tools.compare(conv.numpy(), exp4)
        fc = Linear(
            9,
            3,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        conv = fc(data)
        exp5 = np.array([[0.9998766, 0.9998766, 0.9998766],
                         [0.9998766, 0.9998766, 0.9998766]])
        tools.compare(conv.numpy(), exp5)
        fc = Linear(
            9,
            3,
            param_attr=fluid.initializer.ConstantInitializer(value=0.1),
            act="tanh")
        conv = fc(data)
        exp6 = np.array(
            [[0.716298, 0.716298, 0.716298], [0.716298, 0.716298, 0.716298]])
        tools.compare(conv.numpy(), exp6)
        # print(conv.numpy())

    # def test_fc():
    #     """
    #     test fc
    #     :return:
    #     """
    #     with fluid.dygraph.guard(place):
    #         data = np.ones(shape=(2, 3, 3), dtype=np.float32)
    #         seed = 33
    #         np.random.seed(seed)
    #         fluid.default_startup_program().random_seed = seed
    #         fluid.default_main_program().random_seed = seed
    #
    #         # wx+b w=1 b=1 固定值初始化
    #         fc = FC("fc", 3, param_attr=fluid.initializer.ConstantInitializer(value=1.0),
    #                 bias_attr=fluid.initializer.ConstantInitializer(value=1.0))
    #         data = to_variable(data)
    #         conv = fc(data)
    #         exp1 = np.array([[10., 10., 10.], [10., 10., 10.]])
    #         tools.compare(conv.numpy(), exp1)
    #         # 随机高斯分布初始化
    #         fc = FC("fc", 3, param_attr=fluid.initializer.NormalInitializer(seed=seed))
    #         conv = fc(data)
    #         exp2 = conv.numpy()
    #         fc = FC("fc", 3, param_attr=fluid.initializer.NormalInitializer(seed=seed))
    #         conv = fc(data)
    #         tools.compare(conv.numpy(), exp2)
    #         # 激活函数
    #         fc = FC("fc", 3, param_attr=fluid.initializer.ConstantInitializer(value=-1), act="relu")
    #         conv = fc(data)
    #         exp3 = np.zeros(shape=(2, 3), dtype=np.float32)
    #         tools.compare(conv.numpy(), exp3)
    #         fc = FC("fc", 3, param_attr=fluid.initializer.ConstantInitializer(value=1), act="relu")
    #         conv = fc(data)
    #         exp4 = np.array([[9., 9., 9.], [9., 9., 9.]])
    #         tools.compare(conv.numpy(), exp4)
    #         fc = FC("fc", 3, param_attr=fluid.initializer.ConstantInitializer(value=1), act="sigmoid")
    #         conv = fc(data)
    #         exp5 = np.array([[0.9998766, 0.9998766, 0.9998766], [0.9998766, 0.9998766, 0.9998766]])
    #         tools.compare(conv.numpy(), exp5)
    #         fc = FC("fc", 3, param_attr=fluid.initializer.ConstantInitializer(value=0.1), act="tanh")
    #         conv = fc(data)
    #         exp6 = np.array([[0.716298, 0.716298, 0.716298], [0.716298, 0.716298, 0.716298]])
    #         tools.compare(conv.numpy(), exp6)
    #         # print(conv.numpy())


def test_BackwardStrategy():
    """
    test backward strategy

    Returns: None

    """
    with fluid.dygraph.guard(place):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        x = np.random.random(size=(3, 3)).astype(np.float32)
        inputs2 = []
        for _ in range(10):
            inputs2.append(fluid.dygraph.base.to_variable(x))
        ret2 = fluid.layers.sums(inputs2)
        ret2.stop_gradient = False
        loss2 = fluid.layers.reduce_sum(ret2)
        backward_strategy = fluid.dygraph.BackwardStrategy()
        backward_strategy.sort_sum_gradient = True
        loss2.backward(backward_strategy)
        tools.compare(loss2.gradient(), [1])


def test_conv2d():
    """
    test conv2d

    Returns: None

    """
    with fluid.dygraph.guard(place):
        # 不加激活函数
        data = np.ones(shape=(3, 1, 3, 3), dtype=np.float32)
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        data = to_variable(data)
        conv2d = fluid.dygraph.Conv2D(
            1,
            10,
            2,
            1,
            0,
            param_attr=fluid.initializer.ConstantInitializer(value=1))
        res = conv2d(data)
        exp1 = np.ones(shape=(3, 10, 2, 2), dtype=np.float32) * 4
        tools.compare(res.numpy(), exp1)
        # 加入激活函数relu
        data = np.ones(shape=(3, 1, 3, 3), dtype=np.float32)
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        data = to_variable(data)
        conv2d = fluid.dygraph.Conv2D(
            1,
            10,
            2,
            1,
            0,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            act="relu")
        res = conv2d(data)
        exp2 = np.ones(shape=(3, 10, 2, 2), dtype=np.float32) * 4
        tools.compare(res.numpy(), exp2)
        # 加入激活函数sigmoid
        data = np.ones(shape=(3, 1, 3, 3), dtype=np.float32)
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        data = to_variable(data)
        conv2d = fluid.dygraph.Conv2D(
            1,
            10,
            2,
            1,
            0,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        res = conv2d(data)
        exp3 = np.ones(shape=(3, 10, 2, 2), dtype=np.float32) * tools.sigmoid(4)
        tools.compare(res.numpy(), exp3)
        # 加入bias =1
        data = np.ones(shape=(3, 1, 3, 3), dtype=np.float32)
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        data = to_variable(data)
        conv2d = fluid.dygraph.Conv2D(
            1,
            10,
            2,
            1,
            0,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        res = conv2d(data)
        exp4 = np.ones(shape=(3, 10, 2, 2), dtype=np.float32) * tools.sigmoid(5)
        tools.compare(res.numpy(), exp4)
        # stride = 2
        data = np.ones(shape=(3, 1, 4, 4), dtype=np.float32)
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        data = to_variable(data)
        conv2d = fluid.dygraph.Conv2D(
            1,
            10,
            2,
            2,
            0,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        res = conv2d(data)
        exp4 = np.ones(shape=(3, 10, 2, 2), dtype=np.float32) * tools.sigmoid(5)
        tools.compare(res.numpy(), exp4)
        # padding = 1
        data = np.ones(shape=(3, 1, 1, 1), dtype=np.float32)
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        data = to_variable(data)
        conv2d = fluid.dygraph.Conv2D(
            1,
            10,
            2,
            1,
            1,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        res = conv2d(data)
        exp4 = np.ones(shape=(3, 10, 2, 2), dtype=np.float32) * tools.sigmoid(1)
        tools.compare(res.numpy(), exp4)


def test_conv2dtranspose():
    """
    test conv 2d transpose

    Returns: None
    """
    with fluid.dygraph.guard(place):
        # 不加激活函数
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        data = np.ones(shape=(3, 1, 2, 2), dtype=np.float32)
        data = to_variable(data)
        conv2dtrans = fluid.dygraph.Conv2DTranspose(
            1,
            1,
            filter_size=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1))
        res = conv2dtrans(data)
        exp1 = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1, 1, 2, 1, 2, 4, 2, 1, 2, 1, 1, 2, 1, 2, 4, 2, 1, 2, 1])\
            .reshape(3, 1, 3, 3)
        tools.compare(res.numpy(), exp1)
        # 加入relu
        data = np.ones(shape=(3, 1, 2, 2), dtype=np.float32)
        data = to_variable(data)
        conv2dtrans = fluid.dygraph.Conv2DTranspose(
            1,
            1,
            filter_size=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            act="relu")
        res = conv2dtrans(data)
        exp1 = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1, 1, 2, 1, 2, 4, 2, 1, 2, 1, 1, 2, 1, 2, 4, 2, 1, 2, 1]) \
            .reshape(3, 1, 3, 3)
        tools.compare(res.numpy(), exp1)
        # 加入sigmoid 
        data = np.ones(shape=(3, 1, 2, 2), dtype=np.float32)
        data = to_variable(data)
        conv2dtrans = fluid.dygraph.Conv2DTranspose(
            1,
            1,
            filter_size=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        res = conv2dtrans(data)
        exp1 = np.array([0.880797 ,0.952574 ,0.880797 ,0.952574 ,0.993307 ,0.952574 ,0.880797 ,0.952574 ,0.880797 , \
                        0.880797 ,0.952574 ,0.880797 ,0.952574 ,0.993307 ,0.952574 ,0.880797 ,0.952574 ,0.880797 , \
                        0.880797 ,0.952574 ,0.880797 ,0.952574 ,0.993307 ,0.952574 ,0.880797 ,0.952574 ,0.880797]) \
            .reshape(3, 1, 3, 3)
        tools.compare(res.numpy(), exp1)
        # 加入stride
        data = np.ones(shape=(3, 1, 2, 2), dtype=np.float32)
        data = to_variable(data)
        conv2dtrans = fluid.dygraph.Conv2DTranspose(
            1,
            1,
            filter_size=2,
            stride=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        res = conv2dtrans(data)
        exp1 = np.ones(shape=(3, 1, 4, 4)) * tools.sigmoid(2)
        tools.compare(res.numpy(), exp1)
        # 加入paddding  output size / filter自动识别为3
        data = np.ones(shape=(3, 1, 2, 2), dtype=np.float32)
        data = to_variable(data)
        conv2dtrans = fluid.dygraph.Conv2DTranspose(
            1,
            1,
            filter_size=3,
            padding=1,
            output_size=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1), )
        res = conv2dtrans(data)
        exp1 = np.ones(shape=(3, 1, 2, 2)) * 4
        tools.compare(res.numpy(), exp1)
        # 指定输出大小
        data = np.ones(shape=(3, 1, 2, 2), dtype=np.float32)
        data = to_variable(data)
        conv2dtrans = fluid.dygraph.Conv2DTranspose(
            1,
            1,
            filter_size=2,
            output_size=3,
            param_attr=fluid.initializer.ConstantInitializer(value=1), )
        res = conv2dtrans(data)
        exp1 = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1, 1, 2, 1, 2, 4, 2, 1, 2, 1, 1, 2, 1, 2, 4, 2, 1, 2, 1]) \
            .reshape(3, 1, 3, 3)
        tools.compare(res.numpy(), exp1)


def test_conv3d():
    """
    test conv 3d

    Returns: None
    """
    with fluid.dygraph.guard(place):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        # filter 2*2*2
        data = np.ones(shape=(1, 3, 3, 3, 3), dtype=np.float32)
        data = to_variable(data)
        conv3d = fluid.dygraph.Conv3D(
            3,
            10,
            filter_size=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1))
        res = conv3d(data)
        exp = np.ones(shape=(1, 10, 2, 2, 2), dtype=np.float32) * 24
        tools.compare(res.numpy(), exp)
        # filter 3*2*2
        conv3d = fluid.dygraph.Conv3D(
            3,
            10,
            filter_size=(3, 2, 2),
            param_attr=fluid.initializer.ConstantInitializer(value=1))
        res = conv3d(data)
        exp = np.ones(shape=(1, 10, 1, 2, 2), dtype=np.float32) * 36
        tools.compare(res.numpy(), exp)
        # param = 0.5 bias=1
        data = np.ones(shape=(1, 3, 3, 3, 3), dtype=np.float32)
        data = to_variable(data)
        conv3d = fluid.dygraph.Conv3D(
            3,
            10,
            filter_size=2,
            param_attr=fluid.initializer.ConstantInitializer(value=0.5),
            bias_attr=fluid.initializer.ConstantInitializer(value=1))
        res = conv3d(data)
        exp = np.ones(shape=(1, 10, 2, 2, 2), dtype=np.float32) * 13
        tools.compare(res.numpy(), exp)
        # act = relu
        data = np.ones(shape=(1, 3, 3, 3, 3), dtype=np.float32)
        data = to_variable(data)
        conv3d = fluid.dygraph.Conv3D(
            3,
            10,
            filter_size=2,
            param_attr=fluid.initializer.ConstantInitializer(value=0.5),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            act="relu")
        res = conv3d(data)
        exp = np.ones(shape=(1, 10, 2, 2, 2), dtype=np.float32) * 13
        tools.compare(res.numpy(), exp)
        # act = sigmoid
        data = np.ones(shape=(1, 3, 3, 3, 3), dtype=np.float32)
        data = to_variable(data)
        conv3d = fluid.dygraph.Conv3D(
            3,
            10,
            filter_size=2,
            param_attr=fluid.initializer.ConstantInitializer(value=0.5),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        res = conv3d(data)
        exp = np.ones(
            shape=(1, 10, 2, 2, 2), dtype=np.float32) * tools.sigmoid(13)
        tools.compare(res.numpy(), exp)

        # stride  = 2
        data = np.ones(shape=(1, 3, 4, 4, 4), dtype=np.float32)
        data = to_variable(data)
        conv3d = fluid.dygraph.Conv3D(
            3,
            10,
            filter_size=2,
            stride=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1))
        res = conv3d(data)
        exp = np.ones(shape=(1, 10, 2, 2, 2), dtype=np.float32) * 25
        tools.compare(res.numpy(), exp)

        # padding = 1 6 * 6 * 6 ===> 2 * 2 * 2
        data = np.ones(shape=(1, 3, 4, 4, 4), dtype=np.float32)
        data = to_variable(data)
        conv3d = fluid.dygraph.Conv3D(
            3,
            10,
            filter_size=2,
            stride=2,
            padding=1,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1))
        res = conv3d(data)
        exp = np.array([
            4, 7, 4, 7, 13, 7, 4, 7, 4, 7, 13, 7, 13, 25, 13, 7, 13, 7, 4, 7, 4,
            7, 13, 7, 4, 7, 4, 4, 7, 4, 7, 13, 7, 4, 7, 4, 7, 13, 7, 13, 25, 13,
            7, 13, 7, 4, 7, 4, 7, 13, 7, 4, 7, 4, 4, 7, 4, 7, 13, 7, 4, 7, 4, 7,
            13, 7, 13, 25, 13, 7, 13, 7, 4, 7, 4, 7, 13, 7, 4, 7, 4, 4, 7, 4, 7,
            13, 7, 4, 7, 4, 7, 13, 7, 13, 25, 13, 7, 13, 7, 4, 7, 4, 7, 13, 7,
            4, 7, 4, 4, 7, 4, 7, 13, 7, 4, 7, 4, 7, 13, 7, 13, 25, 13, 7, 13, 7,
            4, 7, 4, 7, 13, 7, 4, 7, 4, 4, 7, 4, 7, 13, 7, 4, 7, 4, 7, 13, 7,
            13, 25, 13, 7, 13, 7, 4, 7, 4, 7, 13, 7, 4, 7, 4, 4, 7, 4, 7, 13, 7,
            4, 7, 4, 7, 13, 7, 13, 25, 13, 7, 13, 7, 4, 7, 4, 7, 13, 7, 4, 7, 4,
            4, 7, 4, 7, 13, 7, 4, 7, 4, 7, 13, 7, 13, 25, 13, 7, 13, 7, 4, 7, 4,
            7, 13, 7, 4, 7, 4, 4, 7, 4, 7, 13, 7, 4, 7, 4, 7, 13, 7, 13, 25, 13,
            7, 13, 7, 4, 7, 4, 7, 13, 7, 4, 7, 4, 4, 7, 4, 7, 13, 7, 4, 7, 4, 7,
            13, 7, 13, 25, 13, 7, 13, 7, 4, 7, 4, 7, 13, 7, 4, 7, 4
        ]).reshape(1, 10, 3, 3, 3)
        tools.compare(res.numpy(), exp)


def test_conv3dtranspose():
    """
    test conv 3d transpose

    Returns: None
    """
    with fluid.dygraph.guard(place):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        # 普通
        data = np.ones(shape=(1, 3, 2, 2, 2), dtype=np.float32)
        data = to_variable(data)
        conv3dtrans = fluid.dygraph.Conv3DTranspose(
            3,
            1,
            filter_size=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1))
        res = conv3dtrans(data)
        exp = np.array([
            3, 6, 3, 6, 12, 6, 3, 6, 3, 6, 12, 6, 12, 24, 12, 6, 12, 6, 3, 6, 3,
            6, 12, 6, 3, 6, 3
        ]).reshape(1, 1, 3, 3, 3)
        tools.compare(res.numpy(), exp)

        # stride = 2
        data = np.ones(shape=(1, 3, 2, 2, 2), dtype=np.float32)
        data = to_variable(data)
        conv3dtrans = fluid.dygraph.Conv3DTranspose(
            3,
            1,
            filter_size=2,
            stride=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1))
        res = conv3dtrans(data)
        exp = np.ones(shape=(1, 1, 4, 4, 4), dtype=np.float32) * 3
        tools.compare(res.numpy(), exp)

        # bias = 1
        data = np.ones(shape=(1, 3, 2, 2, 2), dtype=np.float32)
        data = to_variable(data)
        conv3dtrans = fluid.dygraph.Conv3DTranspose(
            3,
            1,
            filter_size=2,
            stride=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1))
        res = conv3dtrans(data)
        exp = np.ones(shape=(1, 1, 4, 4, 4), dtype=np.float32) * 4
        tools.compare(res.numpy(), exp)

        # act = relu
        data = np.ones(shape=(1, 3, 2, 2, 2), dtype=np.float32)
        data = to_variable(data)
        conv3dtrans = fluid.dygraph.Conv3DTranspose(
            3,
            1,
            filter_size=2,
            stride=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            act="relu")
        res = conv3dtrans(data)
        exp = np.ones(shape=(1, 1, 4, 4, 4), dtype=np.float32) * 4
        tools.compare(res.numpy(), exp)

        # act = sigmoid
        data = np.ones(shape=(1, 3, 2, 2, 2), dtype=np.float32)
        data = to_variable(data)
        conv3dtrans = fluid.dygraph.Conv3DTranspose(
            3,
            1,
            filter_size=2,
            stride=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        res = conv3dtrans(data)
        exp = np.ones(
            shape=(1, 1, 4, 4, 4), dtype=np.float32) * tools.sigmoid(4)
        tools.compare(res.numpy(), exp)

        # padding =1
        data = np.ones(shape=(1, 3, 2, 2, 2), dtype=np.float32)
        data = to_variable(data)
        conv3dtrans = fluid.dygraph.Conv3DTranspose(
            3,
            1,
            filter_size=2,
            stride=2,
            padding=1,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        res = conv3dtrans(data)
        exp = np.ones(
            shape=(1, 1, 2, 2, 2), dtype=np.float32) * tools.sigmoid(4)
        tools.compare(res.numpy(), exp)
        # padding = 2 stride = 3, filter = 3
        data = np.ones(shape=(1, 3, 2, 2, 2), dtype=np.float32)
        data = to_variable(data)
        conv3dtrans = fluid.dygraph.Conv3DTranspose(
            3,
            1,
            filter_size=3,
            stride=3,
            padding=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        res = conv3dtrans(data)
        exp = np.ones(
            shape=(1, 1, 2, 2, 2), dtype=np.float32) * tools.sigmoid(4)
        tools.compare(res.numpy(), exp)


def test_pool2d():
    """
    test pool2d

    Returns: None

    """
    with fluid.dygraph.guard(place):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        # 生成从0 - 15 的数组
        data = np.arange(16.0).reshape(1, 1, 4, 4)
        data = to_variable(data)

        # 基本pool max
        pool2d = fluid.dygraph.Pool2D(pool_size=(2, 2), use_cudnn=False)
        res = pool2d(data)
        exp = np.array([5, 6, 7, 9, 10, 11, 13, 14, 15]).reshape(1, 1, 3, 3)
        tools.compare(res.numpy(), exp)

        # pool avg
        pool2d = fluid.dygraph.Pool2D(
            pool_size=(2, 2), pool_type="avg", use_cudnn=False)
        res = pool2d(data)
        exp = np.array(
            [2.5, 3.5, 4.5, 6.5, 7.5, 8.5, 10.5, 11.5, 12.5]).reshape(1, 1, 3,
                                                                      3)
        tools.compare(res.numpy(), exp)

        # stride = 2
        pool2d = fluid.dygraph.Pool2D(
            pool_size=(2, 2), pool_stride=2, use_cudnn=False)
        res = pool2d(data)
        exp = np.array([5, 7, 13, 15]).reshape(1, 1, 2, 2)
        tools.compare(res.numpy(), exp)

        # stride = 2 pool_type = avg
        pool2d = fluid.dygraph.Pool2D(
            pool_size=(2, 2), pool_stride=2, pool_type="avg", use_cudnn=False)
        res = pool2d(data)
        exp = np.array([2.5, 4.5, 10.5, 12.5]).reshape(1, 1, 2, 2)
        tools.compare(res.numpy(), exp)

        # stride = 3 padding = 1 filter = 3
        pool2d = fluid.dygraph.Pool2D(
            pool_size=(3, 3),
            pool_stride=3,
            pool_type="avg",
            pool_padding=1,
            use_cudnn=False)
        res = pool2d(data)
        exp = np.array([2.5, 4.5, 10.5, 12.5]).reshape(1, 1, 2, 2)
        tools.compare(res.numpy(), exp)

        # stride = 3 padding = 1 filter = 3 pool_type = avg
        pool2d = fluid.dygraph.Pool2D(
            pool_size=(3, 3), pool_stride=3, pool_padding=1, use_cudnn=False)
        res = pool2d(data)
        exp = np.array([5, 7, 13, 15]).reshape(1, 1, 2, 2)
        tools.compare(res.numpy(), exp)

        # stride = 2 padding = 1 filter = 2
        pool2d = fluid.dygraph.Pool2D(
            pool_size=(2, 2), pool_stride=2, pool_padding=1, use_cudnn=False)
        res = pool2d(data)
        exp = np.array([0, 2, 3, 8, 10, 11, 12, 14, 15]).reshape(1, 1, 3, 3)
        tools.compare(res.numpy(), exp)

        # global pooling = True  return the max value
        pool2d = fluid.dygraph.Pool2D(
            pool_size=(2, 2),
            pool_stride=2,
            global_pooling=True,
            pool_padding=1,
            use_cudnn=False)
        res = pool2d(data)
        exp = np.array([15]).reshape(1, 1, 1, 1)
        tools.compare(res.numpy(), exp)

        # test ceil_mode=False
        pool2d = fluid.dygraph.Pool2D(
            pool_size=(2, 2), pool_stride=3, use_cudnn=False, ceil_mode=False)
        res = pool2d(data)
        exp = np.array([5]).reshape(1, 1, 1, 1)
        tools.compare(res.numpy(), exp)

        # test ceil_mode=True
        pool2d = fluid.dygraph.Pool2D(
            pool_size=(2, 2), pool_stride=3, use_cudnn=False, ceil_mode=True)
        res = pool2d(data)
        exp = np.array([5, 7, 13, 15]).reshape(1, 1, 2, 2)
        tools.compare(res.numpy(), exp)

        # exclusive = False
        pool2d = fluid.dygraph.Pool2D(
            pool_size=(3, 3),
            pool_stride=3,
            pool_type="avg",
            pool_padding=1,
            use_cudnn=False,
            exclusive=False)
        res = pool2d(data)
        exp = np.array([1.11111, 2, 4.66667, 5.55556]).reshape(1, 1, 2, 2)
        tools.compare(res.numpy(), exp)

        # pool size = -1
        pool2d = fluid.dygraph.Pool2D(global_pooling=True)
        res = pool2d(data)
        exp = np.array([15]).reshape(1, 1, 1, 1)
        tools.compare(res.numpy(), exp)


def test_BilinearTensorProduct():
    """
    test BilinearTensorProduct

    Returns: None
    """
    with fluid.dygraph.guard(place):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        data = np.ones(shape=(10, 2), dtype=np.float32)
        data = to_variable(data)
        # normal
        btp = fluid.dygraph.BilinearTensorProduct(
            input1_dim=2,
            input2_dim=2,
            output_dim=10,
            param_attr=fluid.initializer.ConstantInitializer(value=1))
        res = btp(data, data)
        exp = np.ones(shape=(10, 10), dtype=np.float32) * 4
        tools.compare(res.numpy(), exp)

        # bias = 1 
        btp = fluid.dygraph.BilinearTensorProduct(
            input1_dim=2,
            input2_dim=2,
            output_dim=10,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1))
        res = btp(data, data)
        exp = np.ones(shape=(10, 10), dtype=np.float32) * 5
        tools.compare(res.numpy(), exp)

        # act = relu
        btp = fluid.dygraph.BilinearTensorProduct(
            input1_dim=2,
            input2_dim=2,
            output_dim=10,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            act="relu")
        res = btp(data, data)
        exp = np.ones(shape=(10, 10), dtype=np.float32) * 4
        tools.compare(res.numpy(), exp)

        # act = sigmoid 
        btp = fluid.dygraph.BilinearTensorProduct(
            input1_dim=2,
            input2_dim=2,
            output_dim=10,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        res = btp(data, data)
        exp = np.ones(shape=(10, 10), dtype=np.float32) * tools.sigmoid(4)
        tools.compare(res.numpy(), exp)


def test_to_variable():
    """
    test to variable

    Returns: None

    """
    with fluid.dygraph.guard(place):
        a = np.ones(shape=(2, 2), dtype=np.float32)
        x = to_variable(a, name="abc")
        tools.compare(x.name, "abc")
        tools.compare(x.numpy(), a)


def test_layer():
    """

    test layer

    Returns: None

    """
    with fluid.dygraph.guard(place):
        a = fluid.dygraph.Layer("test_layer")
        tools.compare(a.full_name(), "test_layer_0")
        a.create_parameter(
            attr=fluid.initializer.ConstantInitializer(value=1),
            shape=(2, 2),
            dtype=np.float32)


def test_Embedding():
    """
    test Embedding

    Returns: None

    """
    inp_word = np.array([[3, 4, 3], [4, 3, 1]]).astype('int64')
    # inp_word = np.expand_dims(inp_word, [-1]).transpose(2, 3, 1)
    dict_size = 20
    with fluid.dygraph.guard(place):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        # basic
        emb = fluid.dygraph.Embedding(
            size=[dict_size, 5],
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            is_sparse=False)
        res = emb(to_variable(inp_word))
        exp = np.ones(shape=[2, 3, 5])
        tools.compare(res.numpy(), exp)
        # is_sparse = True
        emb = fluid.dygraph.Embedding(
            size=[dict_size, 5],
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            is_sparse=True,
            is_distributed=False)
        res = emb(to_variable(inp_word))
        exp = np.ones(shape=[2, 3, 5])
        tools.compare(res.numpy(), exp)
        # random  param
        emb = fluid.dygraph.Embedding(
            size=[dict_size, 5], is_sparse=True, is_distributed=False)
        res = emb(to_variable(inp_word))
        exp = np.array(
            [[[0.37870505, 0.42025283, 0.3309653, 0.45559254, 0.39013025],
              [0.29822686, -0.43142635, 0.17971787, 0.04796883, 0.24556765],
              [0.37870505, 0.42025283, 0.3309653, 0.45559254, 0.39013025]],
             [[0.29822686, -0.43142635, 0.17971787, 0.04796883, 0.24556765],
              [0.37870505, 0.42025283, 0.3309653, 0.45559254, 0.39013025],
              [-0.23538375, 0.48421726, -0.35833406, 0.17234948, 0.03518537]]],
            dtype=np.float32)
        tools.compare(res.numpy(), exp)
        # test good dict_size
        inp_word = np.array(
            [[3, 4, 3, 0], [4, 3, 1, 0], [2, 4, 4, 4]]).astype('int64')
        try:
            dict_size = 5
            emb = fluid.dygraph.Embedding(
                size=[dict_size, 5],
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                is_sparse=False,
                is_distributed=False)
            res = emb(to_variable(inp_word))
            exp = np.ones(shape=[3, 4, 5])
            tools.compare(res.numpy(), exp)

        except:
            print("Embedding Test NOT OK. error in dict_size")
            assert False
        # test overflow dict_size
        inp_word = np.array([[[3, 4, 3, 0], [4, 3, 1, 0], [2, 4, 4, 5]]
                             ]).astype('int64').transpose([1, 2, 0])
        try:
            dict_size = 5
            emb = fluid.dygraph.Embedding(
                size=[dict_size, 5],
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                is_sparse=False,
                is_distributed=False)
            emb(to_variable(inp_word))

        except:
            print("Embedding Test OK. Catch the error in dict_size")
            assert True


def test_TreeConv():
    """
    test Tree Conv

    Returns: None

    """
    with fluid.dygraph.guard(place):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        # basic
        nodes_vector = np.ones(shape=(1, 3, 5)).astype(np.float32)
        edge_set = np.ones(shape=(1, 3, 2)).astype(np.int32)
        treeConv = fluid.dygraph.nn.TreeConv(
            feature_size=5,
            output_size=3,
            num_filters=1,
            max_depth=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            act="relu")
        res = treeConv(
            fluid.dygraph.base.to_variable(nodes_vector),
            fluid.dygraph.base.to_variable(edge_set))
        exp = np.ones(shape=(1, 3, 3, 1)) * 6
        tools.compare(res.numpy(), exp)
        # without biasattr
        nodes_vector = np.ones(shape=(1, 3, 5)).astype(np.float32)
        edge_set = np.ones(shape=(1, 3, 2)).astype(np.int32)
        treeConv = fluid.dygraph.nn.TreeConv(
            feature_size=5,
            output_size=3,
            num_filters=1,
            max_depth=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            act="relu")
        res = treeConv(
            fluid.dygraph.base.to_variable(nodes_vector),
            fluid.dygraph.base.to_variable(edge_set))
        exp = np.ones(shape=(1, 3, 3, 1)) * 5
        tools.compare(res.numpy(), exp)
        # change input nodes vector size
        nodes_vector = np.ones(shape=(1, 3, 4)).astype(np.float32)
        edge_set = np.ones(shape=(1, 3, 2)).astype(np.int32)
        treeConv = fluid.dygraph.nn.TreeConv(
            feature_size=4,
            output_size=3,
            num_filters=1,
            max_depth=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            act="relu")
        res = treeConv(
            fluid.dygraph.base.to_variable(nodes_vector),
            fluid.dygraph.base.to_variable(edge_set))
        exp = np.ones(shape=(1, 3, 3, 1)) * 4
        tools.compare(res.numpy(), exp)
        # sigmoid
        nodes_vector = np.ones(shape=(1, 3, 4)).astype(np.float32)
        edge_set = np.ones(shape=(1, 3, 2)).astype(np.int32)
        treeConv = fluid.dygraph.nn.TreeConv(
            feature_size=4,
            output_size=3,
            num_filters=1,
            max_depth=2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        res = treeConv(
            fluid.dygraph.base.to_variable(nodes_vector),
            fluid.dygraph.base.to_variable(edge_set))
        exp = np.ones(shape=(1, 3, 3, 1)) * tools.sigmoid(4)
        tools.compare(res.numpy(), exp)


def batchnorm_forward(reshape, channel, epsilon, size, data, bias=1, act=None):
    """
    Args:
        reshape(array): reshape
        channel(array): channel
        epsilon(float): epsilon
        size(int): size
        data(array): data
        bias(float): bias
        act(str): act

    Returns:
        result

    """
    data = data.numpy().transpose(1, 0, 2, 3)
    exp = []
    for c in range(channel):
        list = data[c].reshape(size // channel)
        mu = sum(list) / len(list)
        tmp = 0
        for i in list:
            tmp += (mu - i)**2
        sigma = tmp / len(list)
        x = []
        for i in list:
            x.append((i - mu) / math.sqrt(sigma + epsilon))
        y = 1 * np.array(x) + bias
        exp.extend(y)
    if act == "relu":
        for k, v in enumerate(exp):
            exp[k] = v if v > 0 else 0

    elif act == "sigmoid":
        for k, v in enumerate(exp):
            exp[k] = tools.sigmoid(v)

    exp = np.array(exp, dtype=np.float32).reshape(reshape).transpose(1, 0, 2, 3)
    return exp


def test_BatchNorm():
    """
    test BatchNorm

    Returns: None
    """
    with fluid.dygraph.guard(place):
        # batchnorm 是按照channel进行正则化处理的
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        # basic
        channel = 3
        epsilon = 1e-5
        size = 54
        shape = (2, 3, 3, 3)
        reshape = (3, 2, 3, 3)
        data = np.arange(size).astype(np.float32).reshape(shape)
        data = to_variable(data)
        batchnorm = fluid.dygraph.BatchNorm(
            channel,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            epsilon=epsilon)
        res = batchnorm(data)

        # batchnorm算法实现
        # 参数
        exp = batchnorm_forward(reshape, channel, epsilon, size, data)
        tools.compare(res.numpy(), exp)
        # channel = 5 epsilon=1e-7
        channel = 5
        epsilon = 1e-7
        size = 90
        shape = (2, channel, 3, 3)
        reshape = (channel, 2, 3, 3)
        data = np.arange(size).astype(np.float32).reshape(shape)
        data = to_variable(data)
        batchnorm = fluid.dygraph.BatchNorm(
            channel,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            epsilon=epsilon)
        res = batchnorm(data)
        # batchnorm算法实现
        # 参数
        exp = batchnorm_forward(reshape, channel, epsilon, size, data)
        tools.compare(res.numpy(), exp)
        # channel = 1 epsilon=1e-7
        channel = 1
        epsilon = 1e-7
        size = 18
        shape = (2, channel, 3, 3)
        reshape = (channel, 2, 3, 3)
        data = np.arange(size).astype(np.float32).reshape(shape)
        data = to_variable(data)
        batchnorm = fluid.dygraph.BatchNorm(
            channel,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            epsilon=epsilon)
        res = batchnorm(data)
        # batchnorm算法实现
        # 参数
        exp = batchnorm_forward(reshape, channel, epsilon, size, data)
        tools.compare(res.numpy(), exp)
        # channel = 4 epsilon=1e-7 act=relu
        channel = 4
        epsilon = 1e-7
        size = 72
        shape = (2, channel, 3, 3)
        reshape = (channel, 2, 3, 3)
        data = np.arange(size).astype(np.float32).reshape(shape)
        data = to_variable(data)
        batchnorm = fluid.dygraph.BatchNorm(
            channel,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=-0.6),
            epsilon=epsilon,
            act="relu")
        res = batchnorm(data)
        # batchnorm算法实现
        # 参数
        exp = batchnorm_forward(
            reshape, channel, epsilon, size, data, -0.6, act="relu")
        tools.compare(res.numpy(), exp)

        # channel = 4 epsilon=1e-7 act=sigmoid
        channel = 4
        epsilon = 1e-7
        size = 72
        shape = (2, channel, 3, 3)
        reshape = (channel, 2, 3, 3)
        data = np.arange(size).astype(np.float32).reshape(shape)
        data = to_variable(data)
        batchnorm = fluid.dygraph.BatchNorm(
            channel,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=-0.6),
            epsilon=epsilon,
            act="sigmoid")
        res = batchnorm(data)
        # batchnorm算法实现
        # 参数
        exp = batchnorm_forward(
            reshape, channel, epsilon, size, data, -0.6, act="sigmoid")
        tools.compare(res.numpy(), exp)
        # # channel = 4 epsilon=1e-7 act=sigmoid data_layout=NHWC  BUG!!
        # channel = 4
        # epsilon = 1e-7
        # size = 72
        # shape = (2, channel, 3, 3)
        # reshape = (channel, 2, 3, 3)
        # data = np.arange(size).astype(np.float32).reshape(shape).transpose(0, 2, 3, 1)
        # data = to_variable(data)
        # print(data)
        # fluid.layers.batch_norm()
        # batchnorm = fluid.dygraph.BatchNorm("batchnorm", channel,
        #                                     param_attr=fluid.initializer.ConstantInitializer(value=1),
        #                                     bias_attr=fluid.initializer.ConstantInitializer(value=-0.6),
        #                                     epsilon=epsilon,
        #                                     act="sigmoid",
        #                                     data_layout="NHWC")
        # res = batchnorm(data)
        # # batchnorm算法实现
        # # 参数
        # exp = batchnorm_forward(reshape, channel, epsilon, size, data, -0.6, act="sigmoid")
        # tools.compare(res.numpy(), exp)


def test_GroupNorm():
    """
    test GroupNorm

    Returns: None
    """
    with fluid.dygraph.guard(place):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        # basic
        data = np.random.random(size=(3, 3, 3, 3)).astype(np.float32)
        data = to_variable(data)
        groupnorm = fluid.dygraph.GroupNorm(
            3,
            2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1))
        res = groupnorm(data)
        exp = np.array([
            0.216367, 0.861806, 0.736749, 0.254137, 2.20871, 0.013026,
            -0.516801, 2.47416, 1.60018, 0.979102, 2.51189, 0.68055, -0.32491,
            0.546021, -0.0555489, 2.57, 2.2415, 1.00305, 0.914185, 1.09638,
            2.07225, 0.359649, 1.71715, -0.0215964, 0.190991, -0.243247,
            2.91423, -0.462963, 1.5165, 1.9659, 2.86606, 2.32471, 1.25446,
            0.540741, 0.622663, -0.378179, 1.35771, -0.0623521, 0.40883,
            2.39401, 0.653894, 0.57473, 1.92005, -0.550436, 1.05367, 2.0286,
            2.19371, 0.0842438, 0.341804, -0.330516, 1.48994, -0.227736,
            2.30915, 1.11081, -0.370568, 0.829081, 1.65169, 0.0939481, 0.9591,
            -0.264137, 2.05902, 0.834255, 0.656808, 0.847196, -0.522602,
            1.10519, 2.63876, -0.00215995, 2.46265, 0.407517, 2.18933, 2.42491,
            2.17559, 2.50122, -0.00482929, 1.18196, -0.479266, 1.69632,
            0.0205282, 0.369878, 1.5386
        ]).reshape(3, 3, 3, 3)
        tools.compare(exp, res.numpy())
        # act=relu
        data = np.random.random(size=(2, 2, 2, 2)).astype(np.float32)
        data = to_variable(data)
        groupnorm = fluid.dygraph.GroupNorm(
            2,
            2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            act="relu")
        res = groupnorm(data)
        exp = np.array([
            0, 0.625862, 2.27682, 1.50306, 0, 0.452675, 1.02869, 2.59433,
            1.23773, 2.53391, 0.012103, 0.216253, 0, 0.300308, 2.20767, 1.72618
        ]).reshape(2, 2, 2, 2)
        tools.compare(exp, res.numpy())
        # act=sigmoid
        data = np.random.random(size=(2, 2, 2, 2)).astype(np.float32)
        data = to_variable(data)
        groupnorm = fluid.dygraph.GroupNorm(
            2,
            2,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        res = groupnorm(data)
        exp = [
            0.606807, 0.576512, 0.938569, 0.629756, 0.790898, 0.381312,
            0.703176, 0.908143, 0.900439, 0.372695, 0.700639, 0.812787,
            0.852007, 0.901672, 0.476657, 0.531726
        ]
        exp = np.array(exp).reshape(2, 2, 2, 2)
        tools.compare(exp, res.numpy())


def test_LayerNorm():
    """
    test LayerNorm

    Returns: None
    """
    with fluid.dygraph.guard(place):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        # basic
        data = np.random.random(size=(2, 2, 2, 2)).astype(np.float32)
        data = to_variable(data)
        layernorm = fluid.dygraph.LayerNorm(
            [2, 2, 2],
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1))
        res = layernorm(data)
        exp = np.array([
            0.4276, 1.08189, 0.955122, 0.465889, 2.44728, 0.22147, -0.315625,
            2.71637, 1.52833, 0.917017, 2.42569, 0.623161, -0.366484, 0.490748,
            -0.101359, 2.48289
        ]).reshape(2, 2, 2, 2)
        tools.compare(exp, res.numpy())
        # shift = False
        data = np.random.random(size=(2, 2, 2, 2)).astype(np.float32)
        data = to_variable(data)
        layernorm = fluid.dygraph.LayerNorm(
            [2, 2, 2],
            shift=False,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1))
        res = layernorm(data)
        exp = np.array([
            1.6982, -0.00457483, -0.414688, -0.19298, 0.994511, -1.08948,
            0.562406, -1.5534, -0.97359, -1.30768, 1.12161, -1.37865, 0.135875,
            0.479724, 1.16845, 0.754255
        ]).reshape(2, 2, 2, 2)
        tools.compare(exp, res.numpy())
        # shift = False act=sigmoid   激活函数有bug
        data = np.random.random(size=(2, 2, 2, 2)).astype(np.float32)
        data = to_variable(data)
        layernorm = fluid.dygraph.LayerNorm(
            [2, 2, 2],
            shift=False,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            act="sigmoid")
        res = layernorm(data)
        tmp = [
            0.591158, -0.274807, -0.175412, -1.38974, 0.716425, -1.00655,
            -0.434857, 1.97378, -0.385682, -0.47358, 1.02016, -1.72288,
            0.0581959, 1.10575, 1.27404, -0.876004
        ]
        exp = []
        for i in tmp:
            exp.append(tools.sigmoid(i))
        exp = np.array(exp).reshape(2, 2, 2, 2)
        tools.compare(exp, res.numpy())
        # scale = False
        data = np.ones(shape=(2, 2, 2, 2)).astype(np.float32)
        data = to_variable(data)
        layernorm = fluid.dygraph.LayerNorm(
            [2, 2, 2],
            scale=False,
            shift=True,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1))
        res = layernorm(data)
        exp = np.ones(shape=(2, 2, 2, 2)).astype(np.float32)
        tools.compare(exp, res.numpy())


def test_PRelu():
    """
    test_PRelu

    Returns:
        None
    """
    if platform.system() == "Darwin" or platform.system() == "Linux":
        with fluid.dygraph.guard(place):
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            # basic
            data = np.ones(shape=(5, 3, 5, 5), dtype=np.float32)
            data = to_variable(data)
            prelu = fluid.dygraph.PRelu(
                "all",
                param_attr=fluid.initializer.ConstantInitializer(value=1))
            res = prelu(data)
            exp = np.ones(shape=(5, 3, 5, 5), dtype=np.float32)
            tools.compare(res.numpy(), exp)
            # data < 0
            data = np.ones(shape=(5, 3, 5, 5), dtype=np.float32) * -1
            data = to_variable(data)
            prelu = fluid.dygraph.PRelu(
                "all",
                param_attr=fluid.initializer.ConstantInitializer(value=1))
            res = prelu(data)
            exp = np.ones(shape=(5, 3, 5, 5), dtype=np.float32) * -1
            tools.compare(res.numpy(), exp)
            # param = 0.25
            data = np.ones(shape=(5, 3, 5, 5), dtype=np.float32) * -1
            data = to_variable(data)
            prelu = fluid.dygraph.PRelu(
                "all",
                param_attr=fluid.initializer.ConstantInitializer(value=0.25))
            res = prelu(data)
            exp = np.ones(shape=(5, 3, 5, 5), dtype=np.float32) * -0.25
            tools.compare(res.numpy(), exp)
            # mode = channel
            data = np.ones(shape=(5, 3, 5, 5), dtype=np.float32) * -1
            data = to_variable(data)
            prelu = fluid.dygraph.PRelu(
                "channel",
                channel=3,
                param_attr=fluid.initializer.ConstantInitializer(value=0.25))
            res = prelu(data)
            exp = np.ones(shape=(5, 3, 5, 5), dtype=np.float32) * -0.25
            tools.compare(res.numpy(), exp)
            # mode = element
            data = np.ones(shape=(5, 3, 5, 5), dtype=np.float32) * -1
            data = to_variable(data)
            prelu = fluid.dygraph.PRelu(
                "element",
                input_shape=[5, 3, 5, 5],
                param_attr=fluid.initializer.ConstantInitializer(value=0.25))
            res = prelu(data)
            exp = np.ones(shape=(5, 3, 5, 5), dtype=np.float32) * -0.25
            tools.compare(res.numpy(), exp)
            # arange
            data = np.arange(-7, 9).astype(np.float32).reshape(4, 1, 2, 2)
            data = to_variable(data)
            prelu = fluid.dygraph.PRelu(
                "element",
                input_shape=[4, 1, 2, 2],
                param_attr=fluid.initializer.ConstantInitializer(value=1))
            res = prelu(data)
            exp = np.arange(-7, 9).astype(np.float32).reshape(4, 1, 2, 2)
            tools.compare(res.numpy(), exp)
            # arange channel
            data = np.array(
                [-1, -1, -1, -1, -2, -2, -2, -2],
                dtype=np.float32).reshape(1, 2, 2, 2)
            data = to_variable(data)
            prelu = fluid.dygraph.PRelu(
                "channel",
                channel=2,
                param_attr=fluid.initializer.Normal(
                    loc=0.0, scale=1.0))
            res = prelu(data)
            if platform.system() == "Darwin":
                exp = np.array([
                    -1.1513476, -1.1513476, -1.1513476, -1.1513476, 0.51332754,
                    0.51332754, 0.51332754, 0.51332754
                ]).astype(np.float32).reshape(1, 2, 2, 2)
            elif platform.system() == "Linux":
                exp = np.array([
                    0.256664, 0.256664, 0.256664, 0.256664, -2.3027, -2.3027,
                    -2.3027, -2.3027
                ]).astype(np.float32).reshape(1, 2, 2, 2)
            else:
                exp = np.array([
                    0.256664, 0.256664, 0.256664, 0.256664, -2.3027, -2.3027,
                    -2.3027, -2.3027
                ]).astype(np.float32).reshape(1, 2, 2, 2)
            tools.compare(res.numpy(), exp)
            # arange element
            data = np.array(
                [-1, -1, -1, -1, -2, -2, -2, -2],
                dtype=np.float32).reshape(1, 2, 2, 2)
            data = to_variable(data)
            prelu = fluid.dygraph.PRelu(
                "element",
                input_shape=[1, 2, 2, 2],
                param_attr=fluid.initializer.Normal(
                    loc=0.0, scale=1.0))
            res = prelu(data)
            if platform.system() == "Darwin":
                exp = np.array([
                    -1.1513476, 0.25666377, -1.9832053, -0.40487382, -1.2263566,
                    0.01922052, 1.8425357, -1.3875916
                ]).astype(np.float32).reshape(1, 2, 2, 2)
            elif platform.system() == "Linux":
                exp = np.array([
                    0.256664, -1.15135, -0.404874, -1.98321, 0.0192205,
                    -1.22636, -1.38759, 1.84254
                ]).astype(np.float32).reshape(1, 2, 2, 2)
            else:
                exp = np.array([
                    0.256664, -1.15135, -0.404874, -1.98321, 0.0192205,
                    -1.22636, -1.38759, 1.84254
                ]).astype(np.float32).reshape(1, 2, 2, 2)

            tools.compare(res.numpy(), exp)


def test_SpectralNorm():
    """
    test SpectralNorm

    Returns: None
    """
    with fluid.dygraph.guard(place):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        # basic
        data = np.ones(shape=(2, 2, 2, 2)).astype(np.float32)
        data = to_variable(data)
        spectralnorm = fluid.dygraph.SpectralNorm(
            (2, 2, 2, 2), dim=1, power_iters=3)
        res = spectralnorm(data)
        exp = np.ones(shape=(2, 2, 2, 2)).astype(np.float32) * 0.25
        tools.compare(res.numpy(), exp)
        # dim =0
        data = np.ones(shape=(3, 3)).astype(np.float32)
        data = to_variable(data)
        spectralnorm = fluid.dygraph.SpectralNorm((3, 3), dim=0, power_iters=1)
        res = spectralnorm(data)
        exp = np.ones(shape=(3, 3)).astype(np.float32) * 0.333333
        tools.compare(res.numpy(), exp)


def test_GRUUnit():
    """
    test GRU Unit

    Returns: None
    """
    with fluid.dygraph.guard(place):
        seed = 33
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        # basic
        lod = [[1, 1, 1]]
        D = 1
        T = sum(lod[0])
        hidden_input = np.ones(shape=(T, D)).astype('float32')
        x = np.ones(shape=(3, 3)).astype('float32')
        gru = fluid.dygraph.GRUUnit(
            size=D * 3,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            activation="relu",
            gate_activation="relu")
        res = gru(to_variable(x), to_variable(hidden_input))
        exp_hidden = np.array([5, 5, 5]).reshape(3, 1)
        exp_reset_hidden = np.array([2, 2, 2]).reshape(3, 1)
        exp_gate = np.array([2, 2, 3, 2, 2, 3, 2, 2, 3]).reshape(3, 3)
        tools.compare(res[0].numpy(), exp_hidden)
        tools.compare(res[1].numpy(), exp_reset_hidden)
        tools.compare(res[2].numpy(), exp_gate)
        # activation="tanh"
        lod = [[1, 1, 1]]
        D = 1
        T = sum(lod[0])
        hidden_input = np.ones(shape=(T, D)).astype('float32')
        x = np.ones(shape=(3, 3)).astype('float32')
        gru = fluid.dygraph.GRUUnit(
            size=D * 3,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            activation="tanh",
            gate_activation="relu")
        res = gru(to_variable(x), to_variable(hidden_input))
        exp_hidden = np.array([0.990109, 0.990109, 0.990109]).reshape(3, 1)
        exp_reset_hidden = np.array([2, 2, 2]).reshape(3, 1)
        exp_gate = np.array(
            [2, 2, 0.995055, 2, 2, 0.995055, 2, 2, 0.995055]).reshape(3, 3)
        tools.compare(res[0].numpy(), exp_hidden)
        tools.compare(res[1].numpy(), exp_reset_hidden)
        tools.compare(res[2].numpy(), exp_gate)
        # gate_activation="sigmoid"
        lod = [[1, 1, 1]]
        D = 1
        T = sum(lod[0])
        hidden_input = np.ones(shape=(T, D)).astype('float32')
        x = np.ones(shape=(3, 3)).astype('float32')
        gru = fluid.dygraph.GRUUnit(
            size=D * 3,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            activation="relu",
            gate_activation="sigmoid")
        res = gru(to_variable(x), to_variable(hidden_input))
        exp_hidden = np.array([1.7758, 1.7758, 1.7758]).reshape(3, 1)
        exp_reset_hidden = np.array([0.880797, 0.880797, 0.880797]).reshape(3,
                                                                            1)
        exp_gate = np.array([
            0.880797, 0.880797, 1.8808, 0.880797, 0.880797, 1.8808, 0.880797,
            0.880797, 1.8808
        ]).reshape(3, 3)
        tools.compare(res[0].numpy(), exp_hidden)
        tools.compare(res[1].numpy(), exp_reset_hidden)
        tools.compare(res[2].numpy(), exp_gate)
        # bias = 1
        lod = [[1, 1, 1]]
        D = 1
        T = sum(lod[0])
        hidden_input = np.ones(shape=(T, D)).astype('float32')
        x = np.ones(shape=(3, 3)).astype('float32')
        gru = fluid.dygraph.GRUUnit(
            size=D * 3,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            activation="relu",
            gate_activation="relu")
        res = gru(to_variable(x), to_variable(hidden_input))
        exp_hidden = np.array([13, 13, 13]).reshape(3, 1)
        exp_reset_hidden = np.array([3, 3, 3]).reshape(3, 1)
        exp_gate = np.array([3, 3, 5, 3, 3, 5, 3, 3, 5]).reshape(3, 3)
        tools.compare(res[0].numpy(), exp_hidden)
        tools.compare(res[1].numpy(), exp_reset_hidden)
        tools.compare(res[2].numpy(), exp_gate)
        # bias = 1 originmode= True  no use origin mode
        lod = [[1, 1, 1]]
        D = 1
        T = sum(lod[0])
        hidden_input = np.ones(shape=(T, D)).astype('float32')
        x = np.ones(shape=(3, 3)).astype('float32')
        gru = fluid.dygraph.GRUUnit(
            size=D * 3,
            param_attr=fluid.initializer.ConstantInitializer(value=1),
            bias_attr=fluid.initializer.ConstantInitializer(value=1),
            activation="relu",
            gate_activation="relu",
            origin_mode=True)
        res = gru(to_variable(x), to_variable(hidden_input))
        exp_hidden = np.array([13, 13, 13]).reshape(3, 1)
        exp_reset_hidden = np.array([3, 3, 3]).reshape(3, 1)
        exp_gate = np.array([3, 3, 5, 3, 3, 5, 3, 3, 5]).reshape(3, 3)
        tools.compare(res[0].numpy(), exp_hidden)
        tools.compare(res[1].numpy(), exp_reset_hidden)
        tools.compare(res[2].numpy(), exp_gate)


def test_NCE():
    """
    test_NCE

    Returns:
        None
    """
    if platform.system() == "Darwin" or platform.system() == "Linux":
        with fluid.dygraph.guard(place):
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            window_size = 5
            dict_size = 20
            label_word = int(window_size // 2) + 1
            inp_word = np.array([[1], [2], [3], [4], [5]]).astype('int64')
            nid_freq_arr = np.random.dirichlet(np.ones(20) *
                                               1000).astype('float32')
            words = []
            for i in range(window_size):
                words.append(fluid.dygraph.base.to_variable(inp_word[i]))

            emb = fluid.Embedding(
                size=[dict_size, 32], param_attr='emb.w', is_sparse=False)
            embs3 = []
            for i in range(window_size):
                if i == label_word:
                    continue

                emb_rlt = emb(words[i])
                embs3.append(emb_rlt)

            embs3 = fluid.layers.concat(input=embs3, axis=1)
            # basic
            nce = fluid.NCE(dim=embs3.shape[1],
                            num_total_classes=dict_size,
                            num_neg_samples=2,
                            sampler="custom_dist",
                            custom_dist=nid_freq_arr.tolist(),
                            seed=1,
                            param_attr='nce.w',
                            bias_attr='nce.b')

            wl = fluid.layers.unsqueeze(words[label_word], axes=[0])
            nce_loss3 = nce(embs3, wl)
            if platform.system() == "Darwin":
                exp = np.array([[1.2473207]])
            elif platform.system() == "Linux":
                exp = np.array([[1.31181]])
            else:
                exp = np.array([[1.31181]])
            print(nce_loss3)
            tools.compare(nce_loss3.numpy(), exp)
            # num_neg_samples = 5
            nce = fluid.dygraph.NCE(dim=embs3.shape[1],
                                    num_total_classes=dict_size,
                                    num_neg_samples=5,
                                    sampler="custom_dist",
                                    custom_dist=nid_freq_arr.tolist(),
                                    seed=1,
                                    param_attr='nce.w1',
                                    bias_attr='nce.b1')

            nce_loss3 = nce(embs3, wl)
            if platform.system() == "Darwin":
                exp = np.array([[0.9764918]])
            elif platform.system() == "Linux":
                exp = np.array([[1.03003]])
            else:
                exp = np.array([[1.03003]])

            tools.compare(nce_loss3.numpy(), exp)
            # sample_weight = 0 参数丢失
            nce = fluid.dygraph.NCE(dim=embs3.shape[1],
                                    num_total_classes=dict_size,
                                    num_neg_samples=5,
                                    sampler="custom_dist",
                                    custom_dist=nid_freq_arr.tolist(),
                                    seed=1,
                                    sample_weight=0,
                                    param_attr='nce.w2',
                                    bias_attr='nce.b2')

            nce_loss3 = nce(embs3, wl)
            if platform.system() == "Darwin":
                exp = np.array([[0.9764918]])
            elif platform.system() == "Linux":
                exp = np.array([[1.03003]])
            else:
                exp = np.array([[1.03003]])
            tools.compare(nce_loss3.numpy(), exp)
            seed = 33
            nce = fluid.dygraph.NCE(dim=embs3.shape[1],
                                    num_total_classes=dict_size,
                                    num_neg_samples=5,
                                    sampler="custom_dist",
                                    custom_dist=nid_freq_arr.tolist(),
                                    seed=33,
                                    param_attr='nce.w3',
                                    bias_attr='nce.b3')

            nce_loss3 = nce(embs3, wl)
            if platform.system() == "Darwin":
                exp = np.array([[1.0013881]])
            elif platform.system() == "Linux":
                exp = np.array([[0.9668]])
            else:
                exp = np.array([[0.9668]])
            tools.compare(nce_loss3.numpy(), exp)
            # is_sparse = True
            nce = fluid.dygraph.NCE(dim=embs3.shape[1],
                                    num_total_classes=dict_size,
                                    num_neg_samples=5,
                                    sampler="custom_dist",
                                    custom_dist=nid_freq_arr.tolist(),
                                    seed=33,
                                    param_attr='nce.w4',
                                    bias_attr='nce.b4',
                                    is_sparse=True)

            nce_loss3 = nce(embs3, wl)
            if platform.system() == "Darwin":
                exp = np.array([[1.0013881]])
            elif platform.system() == "Linux":
                exp = np.array([[0.9668]])
            else:
                exp = np.array([[0.9668]])
            tools.compare(nce_loss3.numpy(), exp)
            # param
            nce = fluid.dygraph.NCE(
                dim=embs3.shape[1],
                num_total_classes=dict_size,
                num_neg_samples=5,
                sampler="custom_dist",
                custom_dist=nid_freq_arr.tolist(),
                seed=33,
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                bias_attr='nce.b5',
                is_sparse=True)

            nce_loss3 = nce(embs3, wl)
            if platform.system() == "Darwin":
                exp = np.array([[0.9244628]])
            elif platform.system() == "Linux":
                exp = np.array([[0.902682]])
            else:
                exp = np.array([[0.902682]])
            tools.compare(nce_loss3.numpy(), exp)
            # bias
            nce = fluid.dygraph.NCE(
                dim=embs3.shape[1],
                num_total_classes=dict_size,
                num_neg_samples=5,
                sampler="custom_dist",
                custom_dist=nid_freq_arr.tolist(),
                seed=33,
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                bias_attr=fluid.initializer.ConstantInitializer(value=1), )

            nce_loss3 = nce(embs3, wl)
            if platform.system() == "Darwin":
                exp = np.array([[1.1553142]])
            elif platform.system() == "Linux":
                exp = np.array([[1.13023]])
            else:
                exp = np.array([[1.13023]])
            tools.compare(nce_loss3.numpy(), exp)
            # sampler = log_uniform
            nce = fluid.dygraph.NCE(
                dim=embs3.shape[1],
                num_total_classes=dict_size,
                num_neg_samples=5,
                custom_dist=nid_freq_arr.tolist(),
                seed=33,
                sampler="log_uniform",
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                bias_attr=fluid.initializer.ConstantInitializer(value=1), )

            nce_loss3 = nce(embs3, wl)
            exp = np.array([[0.833651]])
            tools.compare(nce_loss3.numpy(), exp)
