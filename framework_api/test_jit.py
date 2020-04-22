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
"""test jit."""

import os
import shutil
os.environ['FLAGS_cudnn_deterministic'] = '1'
import numpy as np
import six
import math
import paddle
import paddle.fluid as fluid
from paddle.fluid import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph import TracedLayer
import tools

batch_size = 16
train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": batch_size,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    },
    "batch_size": batch_size,
    "lr": 0.1,
    "total_images": 1281164,
}


def create_reader(is_test=False):
    """
    define reader
    :param is_test:
    :return:
    """

    def train_reader():
        return paddle.dataset.flowers.train(use_xmap=False)

    def test_reader():
        return paddle.dataset.flowers.test(use_xmap=False)

    reader = train_reader() if not is_test else test_reader()
    return paddle.batch(reader, batch_size=batch_size)


def optimizer_setting(params, parameter_list=None):
    """
    define optimizer
    :param params:
    :param parameter_list:
    :return:
    """
    ls = params["learning_strategy"]
    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        bd = [step * e for e in ls["epochs"]]
        base_lr = params["lr"]
        lr = []
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        if fluid.in_dygraph_mode():
            optimizer = fluid.optimizer.SGD(learning_rate=0.01,
                                            parameter_list=parameter_list)
        else:
            optimizer = fluid.optimizer.SGD(learning_rate=0.01)

    return optimizer


class ConvBNLayer(fluid.Layer):
    """
    define ConvBNLayer
    """

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=None,
            use_cudnn=False)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        """
        forward
        :param inputs:
        :return:
        """
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class BottleneckBlock(fluid.Layer):
    """
    define BottleneckBlock
    """

    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

    def forward(self, inputs):
        """
        forward
        :param inputs:
        :return:
        """
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv2, act='relu')
        return y


class ResNet(fluid.Layer):
    """
    define models
    """

    def __init__(self, layers=50, class_dim=102):
        super(ResNet, self).__init__()

        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_channels = [64, 256, 512, 1024]
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=3, num_filters=64, filter_size=7, stride=2, act='relu')
        self.pool2d_max = Pool2D(
            pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

        self.bottleneck_block_list = []
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels[block]
                        if i == 0 else num_filters[block] * 4,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = Pool2D(
            pool_size=7, pool_type='avg', global_pooling=True)

        self.pool2d_avg_output = num_filters[-1] * 4 * 1 * 1

        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = Linear(
            self.pool2d_avg_output,
            class_dim,
            act='softmax',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        """
        forward
        :param inputs:
        :return:
        """
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.reshape(y, shape=[-1, self.pool2d_avg_output])
        y = self.out(y)
        return y


def train_dygraph(epoch_num):
    """
    train dygraph
    :param epoch_num:
    :return:
    """
    resnet = ResNet()
    optimizer = optimizer_setting(
        train_parameters, parameter_list=resnet.parameters())

    traced_layer = None
    reader = create_reader()

    image_shape = train_parameters['input_size']

    for epoch_id in six.moves.range(epoch_num):
        for i, data in enumerate(reader()):
            image_np = np.array([np.reshape(x[0], image_shape) for x in data])
            label_np = np.array([x[1] for x in data], dtype=np.int64)

            image = to_variable(image_np, zero_copy=False)
            label = to_variable(label_np, zero_copy=False)
            label.stop_gradient = True

            if i == 0 and epoch_id == 0:
                out, traced_layer = TracedLayer.trace(resnet, image)
            else:
                out = resnet(image)

            loss = fluid.layers.cross_entropy(out, label)
            avg_loss = fluid.layers.mean(loss)

            avg_loss.backward()

            optimizer.minimize(avg_loss)

            resnet.clear_gradients()

            if i % 10 == 0:
                print('Epoch {}, batch {}, avg_loss {}'.format(
                    epoch_id, i, avg_loss.numpy()))
    traced_layer.save_inference_model('./infer_dygraph')


def train_static_graph(epoch_num):
    """
    train static graph
    :param epoch_num:
    :return:
    """
    resnet = ResNet()
    image = fluid.data(
        name='image',
        shape=[None] + train_parameters['input_size'],
        dtype='float32')
    label = fluid.data(name='label', shape=[None], dtype='int64')
    out = resnet(image)
    loss = fluid.layers.cross_entropy(out, label)
    avg_loss = fluid.layers.mean(loss)
    optimizer = optimizer_setting(
        train_parameters, parameter_list=resnet.parameters())
    optimizer.minimize(avg_loss)

    program = fluid.default_main_program()
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
    ) else fluid.CPUPlace()
    # place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    image_shape = train_parameters['input_size']

    reader = create_reader()
    for epoch_id in six.moves.range(epoch_num):
        for i, data in enumerate(reader()):
            image_np = np.array([np.reshape(x[0], image_shape) for x in data])
            label_np = np.array([x[1] for x in data], dtype=np.int64)

            avg_loss_val, = exe.run(
                program,
                feed={image.name: image_np,
                      label.name: label_np},
                fetch_list=[avg_loss])
            if i % 10 == 0:
                print('Epoch {}, batch {}, avg_loss {}'.format(epoch_id, i,
                                                               avg_loss_val))

    fluid.io.save_inference_model(
        './infer_static_graph',
        feeded_var_names=[image.name],
        target_vars=[out],
        executor=exe)


def set_random_seed(seed):
    """
    set random seed
    :param seed:
    :return:
    """
    fluid.default_startup_program().random_seed = seed
    fluid.default_main_program().random_seed = seed
    np.random.seed(seed)


def load_static_graph1(epoch_num):
    """
    static graph load dygraph
    :param epoch_num:
    :return:
    """
    np.random.seed(33)
    resnet = ResNet()
    image = fluid.data(
        name='image',
        shape=[None] + train_parameters['input_size'],
        dtype='float32')
    label = fluid.data(name='label', shape=[None], dtype='int64')
    out = resnet(image)
    loss = fluid.layers.cross_entropy(out, label)
    avg_loss = fluid.layers.mean(loss)
    optimizer = optimizer_setting(
        train_parameters, parameter_list=resnet.parameters())
    optimizer.minimize(avg_loss)

    program = fluid.default_main_program()
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
    ) else fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())
    [inference_program, feed_target_names, fetch_targets] = (
        fluid.io.load_inference_model(
            dirname="./infer_dygraph", executor=exe))
    image_shape = train_parameters['input_size']

    reader = create_reader()
    for epoch_id in six.moves.range(epoch_num):
        for i, data in enumerate(reader()):
            image_np = np.array([np.reshape(x[0], image_shape) for x in data])
            label_np = np.array([x[1] for x in data])

            avg_loss_val, = exe.run(inference_program,
                                    feed={feed_target_names[0]: image_np},
                                    fetch_list=fetch_targets)
            print(avg_loss_val.shape)
            return avg_loss_val

            if i % 10 == 0:
                print('Epoch {}, batch {}, avg_loss {}'.format(epoch_id, i,
                                                               avg_loss_val))


def load_static_graph2(epoch_num):
    """
    static graph load static
    :param epoch_num:
    :return:
    """
    np.random.seed(33)
    resnet = ResNet()
    image = fluid.data(
        name='image',
        shape=[None] + train_parameters['input_size'],
        dtype='float32')
    label = fluid.data(name='label', shape=[None], dtype='int64')
    out = resnet(image)
    loss = fluid.layers.cross_entropy(out, label)
    avg_loss = fluid.layers.mean(loss)
    optimizer = optimizer_setting(
        train_parameters, parameter_list=resnet.parameters())
    optimizer.minimize(avg_loss)

    program = fluid.default_main_program()
    place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
    ) else fluid.CPUPlace()
    # place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())
    [inference_program, feed_target_names, fetch_targets] = (
        fluid.io.load_inference_model(
            dirname="./infer_static_graph", executor=exe))
    image_shape = train_parameters['input_size']

    reader = create_reader()
    for epoch_id in six.moves.range(epoch_num):
        for i, data in enumerate(reader()):
            image_np = np.array([np.reshape(x[0], image_shape) for x in data])
            label_np = np.array([x[1] for x in data])

            avg_loss_val, = exe.run(inference_program,
                                    feed={feed_target_names[0]: image_np},
                                    fetch_list=fetch_targets)
            print(avg_loss_val.shape)
            return avg_loss_val


def test_jit():
    """
    test jit
    :return:
    """
    epoch_num = 1
    if os.path.exists("./infer_dygraph"):
        shutil.rmtree("./infer_dygraph")
    if os.path.exists("./infer_static_graph"):
        shutil.rmtree("./infer_static_graph")
    set_random_seed(1)
    train_static_graph(epoch_num)
    with fluid.dygraph.guard():
        set_random_seed(1)
        train_dygraph(epoch_num)
    dygraph = load_static_graph1(1)
    static = load_static_graph2(1)
    for i in range(16):
        for j in range(102):
            if dygraph[i][j] != static[i][j]:
                print("error")
    tools.compare(dygraph, static)
