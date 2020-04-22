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
"""test static nets."""
import numpy as np
import paddle.fluid as fluid
import tools
import paddle
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()


def test_simple_img_conv_pool():
    """
    test simple_img_conv_pool
    :return:
    """
    seed = 33
    np.random.seed(seed)
    fluid.Program().random_seed = seed
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_data = np.ones(shape=(3, 1, 3, 3), dtype=np.float32)
            data = fluid.data(name="data", shape=(3, 1, 3, 3))
            conv2d = fluid.layers.conv2d(
                input=data,
                num_filters=10,
                filter_size=2,
                param_attr=fluid.initializer.ConstantInitializer(value=1))
            pool2d = fluid.layers.pool2d(
                input=conv2d, pool_size=2, pool_stride=1)
            conv_pool = fluid.nets.simple_img_conv_pool(
                input=data,
                num_filters=10,
                filter_size=2,
                pool_size=2,
                pool_stride=1,
                param_attr=fluid.initializer.ConstantInitializer(value=1))
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            res = exe.run(
                fluid.default_main_program(),
                feed={"data": np_data},
                fetch_list=[conv2d.name, pool2d.name, conv_pool.name])
            tools.compare(res[1], res[2])


def test_simple_img_conv_pool_all_para():
    """
    test simple_img_conv_pool
    :return:
    """
    seed = 33
    np.random.seed(seed)
    fluid.Program().random_seed = seed
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_data = np.ones(shape=(3, 1, 3, 3), dtype=np.float32)
            data = fluid.data(name="data", shape=(3, 1, 3, 3))
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            conv2d = fluid.layers.conv2d(
                input=data,
                num_filters=5,
                filter_size=2,
                stride=2,
                padding=1,
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
                act="sigmoid",
                use_cudnn=False,
                dilation=2,
                groups=1, )
            pool2d = fluid.layers.pool2d(
                input=conv2d,
                pool_size=2,
                pool_type='avg',
                pool_stride=2,
                pool_padding=1,
                use_cudnn=False,
                global_pooling=True, )
            conv_pool = fluid.nets.simple_img_conv_pool(
                input=data,
                num_filters=5,
                filter_size=2,
                pool_size=2,
                pool_stride=2,
                pool_padding=1,
                pool_type='avg',
                conv_stride=2,
                conv_padding=1,
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
                act="sigmoid",
                use_cudnn=False,
                global_pooling=True,
                conv_dilation=2,
                conv_groups=1)
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            res = exe.run(
                fluid.default_main_program(),
                feed={"data": np_data},
                fetch_list=[conv2d.name, pool2d.name, conv_pool.name])
            tools.compare(res[1], res[2])


def test_img_conv_group():
    """
    test img_conv_group
    :return:
    """
    seed = 33
    np.random.seed(seed)
    fluid.Program().random_seed = seed
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_data = np.arange(48).astype(np.float32).reshape((1, 3, 4, 4))
            data = fluid.data(name="data", shape=(1, 3, 4, 4))
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            conv2d = fluid.layers.conv2d(
                input=data,
                num_filters=3,
                filter_size=2,
                padding=1,
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                use_cudnn=False, )
            batchnorm = fluid.layers.batch_norm(
                input=conv2d,
                act=None,
                in_place=False, )
            dropout = fluid.layers.dropout(batchnorm, dropout_prob=0.0)
            pool2d_2 = fluid.layers.pool2d(
                input=conv2d, pool_size=2, pool_stride=1)
            conv_group = fluid.nets.img_conv_group(
                input=data,
                conv_num_filter=[3],
                pool_size=2,
                conv_padding=1,
                conv_filter_size=2,
                conv_act=None,
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                conv_with_batchnorm=False,
                conv_batchnorm_drop_rate=0.0,
                pool_stride=1,
                pool_type="max",
                use_cudnn=False, )
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            res = exe.run(fluid.default_main_program(),
                          feed={"data": np_data},
                          fetch_list=[
                              conv2d.name, batchnorm.name, dropout,
                              pool2d_2.name, conv_group.name
                          ])
            tools.compare(res[3], res[4])


def test_img_conv_group_all_para():
    """
    test img_conv_group
    :return:
    """
    seed = 33
    np.random.seed(seed)
    fluid.Program().random_seed = seed
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_data = np.arange(144).astype(np.float32).reshape((3, 3, 4, 4))
            data = fluid.data(name="data", shape=(3, 3, 4, 4))
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            conv2d = fluid.layers.conv2d(
                input=data,
                num_filters=2,
                filter_size=3,
                padding=1,
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                use_cudnn=False, )
            batchnorm = fluid.layers.batch_norm(
                input=conv2d,
                act="sigmoid",
                in_place=False, )
            dropout = fluid.layers.dropout(batchnorm, dropout_prob=0.5)
            pool2d = fluid.layers.pool2d(
                input=dropout, pool_size=3, pool_stride=1, pool_type='avg')
            conv_group = fluid.nets.img_conv_group(
                input=data,
                conv_num_filter=[2],
                pool_size=3,
                conv_padding=1,
                conv_filter_size=3,
                conv_act="sigmoid",
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                conv_with_batchnorm=True,
                conv_batchnorm_drop_rate=0.5,
                pool_stride=1,
                pool_type="avg",
                use_cudnn=False, )
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            res = exe.run(fluid.default_main_program(),
                          feed={"data": np_data},
                          fetch_list=[
                              conv2d.name, batchnorm.name, dropout, pool2d.name,
                              conv_group.name
                          ])
            tools.compare(res[3], res[4])


def test_glu():
    """
    test glu
    :return:
    """
    seed = 33
    np.random.seed(seed)
    fluid.Program().random_seed
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_data = np.arange(144).reshape([-1, 6, 3, 8]).astype("float32")
            data = fluid.data(name="data", shape=[-1, 6, 3, 8], dtype="float32")
            split_1, split_2 = fluid.layers.split(
                input=data, num_or_sections=2, dim=-1)
            sigmoid_2 = fluid.layers.sigmoid(x=split_2)
            mul = fluid.layers.elementwise_mul(x=split_1, y=sigmoid_2)
            output = fluid.nets.glu(input=data, dim=-1)
            exe = fluid.Executor(place=fluid.CPUPlace())
            res = exe.run(
                fluid.default_main_program(),
                feed={"data": np_data},
                fetch_list=[
                    # split_1.name,
                    # split_2.name,
                    # sigmoid_2,
                    mul.name,
                    output.name
                ])
            tools.compare(res[0], res[1])


def test_glu_dim1():
    """
    test_glu_dim1
    :return:
    """
    seed = 33
    np.random.seed(seed)
    fluid.Program().random_seed
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_data = np.arange(16).reshape([-1, 2, 2, 2]).astype("float32")
            data = fluid.data(name="data", shape=[-1, 2, 2, 2], dtype="float32")
            split_1, split_2 = fluid.layers.split(
                input=data, num_or_sections=2, dim=1)
            sigmoid_2 = fluid.layers.sigmoid(x=split_2)
            mul = fluid.layers.elementwise_mul(x=split_1, y=sigmoid_2)
            output = fluid.nets.glu(input=data, dim=1)
            exe = fluid.Executor(place=fluid.CPUPlace())
            res = exe.run(
                fluid.default_main_program(),
                feed={"data": np_data},
                fetch_list=[
                    # split_1.name,
                    # split_2.name,
                    # sigmoid_2,
                    mul.name,
                    output.name
                ])
            tools.compare(res[0], res[1])


def test_glu_dim2():
    """
    test_glu_dim2
    :return:
    """
    seed = 33
    np.random.seed(seed)
    fluid.Program().random_seed
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_data = np.arange(16).reshape([-1, 2, 2, 2]).astype("float32")
            data = fluid.data(name="data", shape=[-1, 2, 2, 2], dtype="float32")
            split_1, split_2 = fluid.layers.split(
                input=data, num_or_sections=2, dim=2)
            sigmoid_2 = fluid.layers.sigmoid(x=split_2)
            mul = fluid.layers.elementwise_mul(x=split_1, y=sigmoid_2)
            output = fluid.nets.glu(input=data, dim=2)
            exe = fluid.Executor(place=fluid.CPUPlace())
            res = exe.run(
                fluid.default_main_program(),
                feed={"data": np_data},
                fetch_list=[
                    # split_1.name,
                    # split_2.name,
                    # sigmoid_2,
                    mul.name,
                    output.name
                ])
            tools.compare(res[0], res[1])


def test_glu_dim3():
    """
    test_glu_dim3
    :return:
    """
    seed = 33
    np.random.seed(seed)
    fluid.Program().random_seed
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_data = np.arange(16).reshape([-1, 2, 2, 2]).astype("float32")
            data = fluid.data(name="data", shape=[-1, 2, 2, 2], dtype="float32")
            split_1, split_2 = fluid.layers.split(
                input=data, num_or_sections=2, dim=3)
            sigmoid_2 = fluid.layers.sigmoid(x=split_2)
            mul = fluid.layers.elementwise_mul(x=split_1, y=sigmoid_2)
            output = fluid.nets.glu(input=data, dim=3)
            exe = fluid.Executor(place)
            res = exe.run(
                fluid.default_main_program(),
                feed={"data": np_data},
                fetch_list=[
                    # split_1.name,
                    # split_2.name,
                    # sigmoid_2,
                    mul.name,
                    output.name
                ])
            tools.compare(res[0], res[1])


def test_glu_LoD():
    """
    test_glu_LoD
    :return:
    """
    seed = 33
    np.random.seed(seed)
    fluid.Program().random_seed
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_data = np.arange(144).reshape([-1, 6, 3, 8]).astype("float32")
            data = fluid.data(
                name="data", shape=[-1, 6, 3, 8], dtype="float32", lod_level=1)
            split_1, split_2 = fluid.layers.split(
                input=data, num_or_sections=2, dim=-1)
            sigmoid_2 = fluid.layers.sigmoid(x=split_2)
            mul = fluid.layers.elementwise_mul(x=split_1, y=sigmoid_2)
            output = fluid.nets.glu(input=data, dim=-1)
            exe = fluid.Executor(place)
            res = exe.run(
                fluid.default_main_program(),
                feed={"data": np_data},
                fetch_list=[
                    # split_1.name,
                    # split_2.name,
                    # sigmoid_2,
                    mul.name,
                    output.name
                ])
            tools.compare(res[0], res[1])


def test_glu_LoD_dim1():
    """
    test_glu_LoD_dim1
    :return:
    """
    seed = 33
    np.random.seed(seed)
    fluid.Program().random_seed
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            np_data = np.arange(18).reshape([-1, 2, 3, 3]).astype("float32")
            data = fluid.data(
                name="data", shape=[-1, 2, 3, 3], dtype="float32", lod_level=1)
            split_1, split_2 = fluid.layers.split(
                input=data, num_or_sections=2, dim=1)
            sigmoid_2 = fluid.layers.sigmoid(x=split_2)
            mul = fluid.layers.elementwise_mul(x=split_1, y=sigmoid_2)
            output = fluid.nets.glu(input=data, dim=1)
            exe = fluid.Executor(place)
            res = exe.run(
                fluid.default_main_program(),
                feed={"data": np_data},
                fetch_list=[
                    # split_1.name,
                    # split_2.name,
                    # sigmoid_2,
                    mul.name,
                    output.name
                ])
            tools.compare(res[0], res[1])


def test_sequence_conv_pool():
    """
    test sequence_conv_pool
    :return:
    """
    seed = 33
    np.random.seed(seed)
    fluid.Program().random_seed = seed
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            # np_data = np.arange(4).reshape([2, 2]).astype("float32")
            # print(np_data)
            np_data = fluid.LoDTensor()
            np_data.set(np.arange(4).reshape([2, 2]).astype("float32"), place)
            # print(np_data)
            np_data.set_lod([[0, 2, 2]])
            # print(np_data)

            # data = fluid.layers.data(name="data", shape=[2, 2], dtype="float32", lod_level=1)
            # print(data)
            data = fluid.data(
                name="data", shape=[2, 2], dtype="float32", lod_level=1)
            # print(data)
            sequence_conv = fluid.layers.sequence_conv(
                input=data,
                num_filters=3,
                filter_size=3,
                filter_stride=1,
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                act="sigmoid")
            sequence_pool = fluid.layers.sequence_pool(
                input=sequence_conv, pool_type='max')
            conv_pool = fluid.nets.sequence_conv_pool(
                input=data,
                num_filters=3,
                filter_size=3,
                param_attr=fluid.initializer.ConstantInitializer(value=1))

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            res = exe.run(fluid.default_main_program(),
                          feed={"data": np_data},
                          return_numpy=False,
                          fetch_list=[
                              sequence_conv.name,
                              sequence_pool.name,
                              conv_pool.name,
                          ])
            tools.compare(np.array(res[1]), np.array(res[2]))


def test_sequence_conv_pool():
    """
    test sequence_conv_pool_all_para
    :return:
    """
    seed = 33
    np.random.seed(seed)
    fluid.Program().random_seed = seed
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            # np_data = np.arange(4).reshape([2, 2]).astype("float32")
            # print(np_data)
            np_data = fluid.LoDTensor()
            np_data.set(np.arange(10).reshape([2, 5]).astype("float32"), place)
            # print(np_data)
            np_data.set_lod([[0, 2, 2]])
            # print(np_data)

            # data = fluid.layers.data(name="data", shape=[2, 2], dtype="float32", lod_level=1)
            data = fluid.data(
                name="data", shape=[2, 5], dtype="float32", lod_level=1)
            sequence_conv = fluid.layers.sequence_conv(
                input=data,
                num_filters=3,
                filter_size=2,
                filter_stride=1,
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
                act="tanh")
            sequence_pool = fluid.layers.sequence_pool(
                input=sequence_conv,
                pool_type="sqrt", )
            conv_pool = fluid.nets.sequence_conv_pool(
                input=data,
                num_filters=3,
                filter_size=2,
                act="tanh",
                pool_type="sqrt",
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                bias_attr=fluid.initializer.ConstantInitializer(value=1), )

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            res = exe.run(fluid.default_main_program(),
                          feed={"data": np_data},
                          return_numpy=False,
                          fetch_list=[
                              sequence_conv.name,
                              sequence_pool.name,
                              conv_pool.name,
                          ])
            tools.compare(np.array(res[1]), np.array(res[2]))
