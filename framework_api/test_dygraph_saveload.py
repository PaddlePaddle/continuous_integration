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
"""test dygraph saveload."""

from __future__ import print_function

import numpy as np
from PIL import Image
from nose import tools
import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import SGDOptimizer, AdamOptimizer
from paddle.fluid.dygraph.base import to_variable
from mnist import MNIST
import cts_tools


def train_in_cpu_infer_in_cpu():
    """
    cpu 下训练  cpu预测
    :return:
    """
    seed = 90
    epoch_num = 1
    BATCH_SIZE = 64
    with fluid.dygraph.guard(fluid.CPUPlace()):
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        mnist = MNIST()
        adam = AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost = mnist(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))
            mnist.eval()

            def load_image(file):
                im = Image.open(file).convert('L')
                im = im.resize((28, 28), Image.ANTIALIAS)
                im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
                im = im / 255.0 * 2.0 - 1.0
                return im

            cur_dir = os.path.dirname(os.path.realpath(__file__))
            tensor_img = load_image(cur_dir + '/image/mnist/2.png')
            results = mnist(to_variable(tensor_img))
            return results.numpy()[0]


def train_in_gpu_infer_in_cpu():
    """
    gpu 下训练  cpu预测
    :return:
    """
    seed = 90
    epoch_num = 1
    BATCH_SIZE = 64
    # np.set_printoptions(precision=3, suppress=True)
    diff_var_save = {}
    diff_var_load = {}
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        mnist = MNIST()
        adam = AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost = mnist(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))
        fluid.save_dygraph(mnist.state_dict(), "save_dir")
        fluid.save_dygraph(adam.state_dict(), "save_dir")
        for key_i, key_bo in mnist.state_dict().items():
            diff_var_save[key_bo.name] = key_bo.numpy()
        print("checkpoint saved")

    with fluid.dygraph.guard(fluid.CPUPlace()):
        mnist_infer = MNIST()
        mnist_infer.eval()
        adam = AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())
        # load checkpoint
        para_state_dict, opti_state_dict = fluid.load_dygraph( "save_dir")
        mnist_infer.set_dict(para_state_dict)
        adam.set_dict(opti_state_dict)
        print(adam._learning_rate)
        print("checkpoint loaded")

        # start evaluate mode
        def load_image(file):
            im = Image.open(file).convert('L')
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            return im

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        tensor_img = load_image(cur_dir + '/image/mnist/2.png')
        results = mnist_infer(to_variable(tensor_img))
        os.remove("save_dir.pdparams")
        os.remove("save_dir.pdopt")
        return results.numpy()[0]


def train_in_cpu_infer_in_gpu():
    """
    cpu 下训练  gpu预测
    :return:
    """
    seed = 90
    epoch_num = 1
    BATCH_SIZE = 64
    # np.set_printoptions(precision=3, suppress=True)
    diff_var_save = {}
    diff_var_load = {}
    with fluid.dygraph.guard(fluid.CPUPlace()):
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        mnist = MNIST()
        adam = AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost = mnist(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))
        fluid.save_dygraph(mnist.state_dict(), "save_dir")
        for key_i, key_bo in mnist.state_dict().items():
            diff_var_save[key_bo.name] = key_bo.numpy()
        print("checkpoint saved")

    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        mnist_infer = MNIST()
        mnist_infer.eval()
        # load checkpoint
        para_state_dict, opti_state_dict = fluid.load_dygraph("save_dir")
        mnist_infer.set_dict(para_state_dict)
        print(adam._learning_rate)
        print("checkpoint loaded")

        # start evaluate mode
        def load_image(file):
            im = Image.open(file).convert('L')
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            return im

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        tensor_img = load_image(cur_dir + '/image/mnist/2.png')
        results = mnist_infer(to_variable(tensor_img))
        os.remove("save_dir.pdparams")
        return results.numpy()[0]


def train_in_gpu_infer_in_gpu():
    """
    gpu 下训练  gpu预测
    :return:
    """
    seed = 90
    epoch_num = 1
    BATCH_SIZE = 64
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        mnist = MNIST()
        adam = AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost = mnist(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))
            mnist.eval()

            # test_cost, test_acc = self._test_train(test_reader, mnist, BATCH_SIZE)
            # mnist.train()
            # print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(epoch, test_cost, test_acc))

            def load_image(file):
                im = Image.open(file).convert('L')
                im = im.resize((28, 28), Image.ANTIALIAS)
                im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
                im = im / 255.0 * 2.0 - 1.0
                return im

            cur_dir = os.path.dirname(os.path.realpath(__file__))
            tensor_img = load_image(cur_dir + '/image/mnist/2.png')
            results = mnist(to_variable(tensor_img))
            return results.numpy()[0]


def save_in_cpu_load_in_cpu():
    """
    cpu 下训练  save  cpu load
    :return:
    """
    seed = 90
    epoch_num = 1
    BATCH_SIZE = 64
    # np.set_printoptions(precision=3, suppress=True)
    diff_var_save = {}
    diff_var_load = {}
    with fluid.dygraph.guard(fluid.CPUPlace()):
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        mnist = MNIST()
        adam = AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost = mnist(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))
        fluid.save_dygraph(mnist.state_dict(), "save_dir")
        fluid.save_dygraph(adam.state_dict(), "save_dir")
        for key_i, key_bo in mnist.state_dict().items():
            diff_var_save[key_bo.name] = key_bo.numpy()
        print("checkpoint saved")

    with fluid.dygraph.guard(fluid.CPUPlace()):
        mnist_infer = MNIST()
        mnist_infer.eval()
        adam = AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())
        # load checkpoint
        para_state_dict, opti_state_dict = fluid.load_dygraph("save_dir")
        mnist_infer.set_dict(para_state_dict)
        adam.set_dict(opti_state_dict)
        print(adam._learning_rate)
        print("checkpoint loaded")

        # start evaluate mode
        def load_image(file):
            im = Image.open(file).convert('L')
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            return im

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        tensor_img = load_image(cur_dir + '/image/mnist/2.png')
        results = mnist_infer(to_variable(tensor_img))
        os.remove("save_dir.pdparams")
        os.remove("save_dir.pdopt")
        return results.numpy()[0]


def save_in_gpu_load_in_gpu():
    """
    gpu 下训练  save  gpu load
    :return:
    """
    seed = 90
    epoch_num = 1
    BATCH_SIZE = 64
    # np.set_printoptions(precision=3, suppress=True)
    diff_var_save = {}
    diff_var_load = {}
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        mnist = MNIST()
        adam = AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost = mnist(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))
        fluid.save_dygraph(mnist.state_dict(), "save_dir")
        fluid.save_dygraph(adam.state_dict(), "save_dir")
        for key_i, key_bo in mnist.state_dict().items():
            diff_var_save[key_bo.name] = key_bo.numpy()
        print("checkpoint saved")

    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        mnist_infer = MNIST()
        mnist_infer.eval()
        adam = AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())
        # load checkpoint
        para_state_dict, opti_state_dict = fluid.load_dygraph("save_dir")
        mnist_infer.set_dict(para_state_dict)
        adam.set_dict(opti_state_dict)
        print(adam._learning_rate)
        print("checkpoint loaded")

        # start evaluate mode
        def load_image(file):
            im = Image.open(file).convert('L')
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            return im

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        tensor_img = load_image(cur_dir + '/image/mnist/2.png')
        results = mnist_infer(to_variable(tensor_img))
        os.remove("save_dir.pdparams")
        os.remove("save_dir.pdopt")
        return results.numpy()[0]


def save_in_cpu_load_in_cpu_with_lrdecay():
    """
    cpu 下训练  cpu预测  lrdecay
    :return:
    """
    seed = 90
    epoch_num = 1
    BATCH_SIZE = 64
    # np.set_printoptions(precision=3, suppress=True)
    diff_var_save = {}
    diff_var_load = {}
    with fluid.dygraph.guard(fluid.CPUPlace()):
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        mnist = MNIST()
        adam = AdamOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                    learning_rate=0.01,
                    decay_steps=100,
                    decay_rate=0.8,
                    staircase=True), parameter_list=mnist.parameters())
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost = mnist(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))
        fluid.save_dygraph(mnist.state_dict(), "save_dir")
        fluid.save_dygraph(adam.state_dict(), "save_dir")
        for key_i, key_bo in mnist.state_dict().items():
            diff_var_save[key_bo.name] = key_bo.numpy()
        print("checkpoint saved")

    with fluid.dygraph.guard(fluid.CPUPlace()):
        mnist_infer = MNIST()
        mnist_infer.eval()
        adam = AdamOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                    learning_rate=0.01,
                    decay_steps=100,
                    decay_rate=0.8,
                    staircase=True), parameter_list=mnist.parameters())
        # load checkpoint
        para_state_dict, opti_state_dict = fluid.load_dygraph("save_dir")
        mnist_infer.set_dict(para_state_dict)
        adam.set_dict(opti_state_dict)
        print(adam._learning_rate)
        assert adam._accumulators_holder["global_step"][0] == 937
        print("checkpoint loaded")

        # start evaluate mode
        def load_image(file):
            im = Image.open(file).convert('L')
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            return im

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        tensor_img = load_image(cur_dir + '/image/mnist/2.png')
        results = mnist_infer(to_variable(tensor_img))
        os.remove("save_dir.pdparams")
        os.remove("save_dir.pdopt")
        return results.numpy()[0]


def save_in_cpu_load_in_gpu_with_lrdecay():
    """
    cpu 下训练  gpu预测  lrdecay
    :return:
    """
    seed = 90
    epoch_num = 1
    BATCH_SIZE = 64
    # np.set_printoptions(precision=3, suppress=True)
    diff_var_save = {}
    diff_var_load = {}
    with fluid.dygraph.guard(fluid.CPUPlace()):
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        mnist = MNIST()
        adam = AdamOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                    learning_rate=0.01,
                    decay_steps=100,
                    decay_rate=0.8,
                    staircase=True), parameter_list=mnist.parameters())
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost = mnist(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))
        fluid.save_dygraph(mnist.state_dict(), "save_dir")
        fluid.save_dygraph(adam.state_dict(), "save_dir")
        for key_i, key_bo in mnist.state_dict().items():
            diff_var_save[key_bo.name] = key_bo.numpy()
        print("checkpoint saved")

    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        mnist_infer = MNIST()
        mnist_infer.eval()
        adam = AdamOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                    learning_rate=0.01,
                    decay_steps=100,
                    decay_rate=0.8,
                    staircase=True), parameter_list=mnist.parameters())
        # load checkpoint
        para_state_dict, opti_state_dict = fluid.load_dygraph("save_dir")
        mnist_infer.set_dict(para_state_dict)
        adam.set_dict(opti_state_dict)
        print(adam._learning_rate)
        assert adam._accumulators_holder["global_step"][0] == 937
        print("checkpoint loaded")

        # start evaluate mode
        def load_image(file):
            im = Image.open(file).convert('L')
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            return im

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        tensor_img = load_image(cur_dir + '/image/mnist/2.png')
        results = mnist_infer(to_variable(tensor_img))
        os.remove("save_dir.pdparams")
        os.remove("save_dir.pdopt")
        return results.numpy()[0]


def save_in_gpu_load_in_cpu_with_lrdecay():
    """
    gpu 下训练  cpu预测  lrdecay
    :return:
    """
    seed = 90
    epoch_num = 1
    BATCH_SIZE = 64
    # np.set_printoptions(precision=3, suppress=True)
    diff_var_save = {}
    diff_var_load = {}
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        mnist = MNIST()
        adam = AdamOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                    learning_rate=0.01,
                    decay_steps=100,
                    decay_rate=0.8,
                    staircase=True), parameter_list=mnist.parameters())
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost = mnist(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))
        fluid.save_dygraph(mnist.state_dict(), "save_dir")
        fluid.save_dygraph(adam.state_dict(), "save_dir")
        for key_i, key_bo in mnist.state_dict().items():
            diff_var_save[key_bo.name] = key_bo.numpy()
        print("checkpoint saved")

    with fluid.dygraph.guard(fluid.CPUPlace()):
        mnist_infer = MNIST()
        mnist_infer.eval()
        adam = AdamOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                    learning_rate=0.01,
                    decay_steps=100,
                    decay_rate=0.8,
                    staircase=True), parameter_list=mnist.parameters())
        # load checkpoint
        para_state_dict, opti_state_dict = fluid.load_dygraph("save_dir")
        mnist_infer.set_dict(para_state_dict)
        adam.set_dict(opti_state_dict)
        print(adam._learning_rate)
        assert adam._accumulators_holder["global_step"][0] == 937
        print("checkpoint loaded")

        # start evaluate mode
        def load_image(file):
            im = Image.open(file).convert('L')
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            return im

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        tensor_img = load_image(cur_dir + '/image/mnist/2.png')
        results = mnist_infer(to_variable(tensor_img))
        os.remove("save_dir.pdparams")
        os.remove("save_dir.pdopt")
        return results.numpy()[0]


def save_in_gpu_load_in_gpu_with_lrdecay():
    """
    gpu 下训练  gpu预测  lrdecay
    :return:
    """
    seed = 90
    epoch_num = 1
    BATCH_SIZE = 64
    # np.set_printoptions(precision=3, suppress=True)
    diff_var_save = {}
    diff_var_load = {}
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed

        mnist = MNIST()
        adam = AdamOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                    learning_rate=0.01,
                    decay_steps=100,
                    decay_rate=0.8,
                    staircase=True), parameter_list=mnist.parameters())
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28)
                     for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost = mnist(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(epoch, batch_id, avg_loss.numpy()))
        fluid.save_dygraph(mnist.state_dict(), "save_dir")
        fluid.save_dygraph(adam.state_dict(), "save_dir")
        for key_i, key_bo in mnist.state_dict().items():
            diff_var_save[key_bo.name] = key_bo.numpy()
        print("checkpoint saved")

    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        mnist_infer = MNIST()
        mnist_infer.eval()
        adam = AdamOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                    learning_rate=0.01,
                    decay_steps=100,
                    decay_rate=0.8,
                    staircase=True), parameter_list=mnist.parameters())
        # load checkpoint
        para_state_dict, opti_state_dict = fluid.load_dygraph("save_dir")
        mnist_infer.set_dict(para_state_dict)
        adam.set_dict(opti_state_dict)
        print(adam._learning_rate)
        assert adam._accumulators_holder["global_step"][0] == 937
        print("checkpoint loaded")

        # start evaluate mode
        def load_image(file):
            im = Image.open(file).convert('L')
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            return im

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        tensor_img = load_image(cur_dir + '/image/mnist/2.png')
        results = mnist_infer(to_variable(tensor_img))
        os.remove("save_dir.pdparams")
        os.remove("save_dir.pdopt")
        return results.numpy()[0]


def load_wrongmodel_with_lrdecay():
    """
    gpu 下训练  gpu预测  lrdecay
    :return:
    """
    seed = 90
    epoch_num = 1
    BATCH_SIZE = 64
    # np.set_printoptions(precision=3, suppress=True)
    diff_var_save = {}
    diff_var_load = {}

    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        mnist_infer = MNIST()
        mnist_infer.eval()
        adam = AdamOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                    learning_rate=0.01,
                    decay_steps=50,
                    decay_rate=0.001,
                    staircase=True), parameter_list=mnist_infer.parameters())
        # load checkpoint
        try:
            para_state_dict, opti_state_dict = fluid.load_dygraph("wrong")
            mnist_infer.set_dict(para_state_dict)
            adam.set_dict(opti_state_dict)
            print(adam._learning_rate)
            assert adam._accumulators_holder["global_step"][0] == 937
            print("checkpoint loaded")

            # start evaluate mode
            def load_image(file):
                im = Image.open(file).convert('L')
                im = im.resize((28, 28), Image.ANTIALIAS)
                im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
                im = im / 255.0 * 2.0 - 1.0
                return im

            cur_dir = os.path.dirname(os.path.realpath(__file__))
            tensor_img = load_image(cur_dir + '/image/mnist/2.png')
            results = mnist_infer(to_variable(tensor_img))
            os.remove("save_dir.pdparams")
            os.remove("save_dir.pdopt")
            return results.numpy()[0]
        except RuntimeError:
            assert True


class TestDifferentPlaceTrainAndInfer(object):
    """
    验证不同place下 mnist的训练和预测
    """
    def __init__(self):
        self.cpu_infer = [5.7926900e-07, 1.0607594e-04, 9.9969727e-01,
        1.9561472e-04, 4.8384036e-10, 2.9619747e-08, 3.9836408e-08,
        2.1984442e-11, 4.6068348e-07, 4.7721054e-09]
        self.gpu_infer = [2.9659088e-06, 3.3285178e-05, 9.9736077e-01,
        2.6023386e-03, 5.7552263e-10, 1.3914855e-07, 3.8529805e-08,
        3.1219374e-11, 4.3815280e-07, 3.9480302e-08]
        self.delta = 1e-3
        self.cpu_decay = [5.8745218e-06, 1.9231631e-04, 9.9442106e-01,
                          5.3762482e-03, 3.0175872e-11, 1.9730437e-06,
                          1.7275872e-06, 1.5791408e-10, 6.7389209e-07, 1.4677698e-09]
        self.gpu_decay = [6.3934053e-06, 5.8440792e-06, 9.9852836e-01,
                          1.4529644e-03, 9.4327185e-11, 2.1644519e-07,
                          6.2045478e-06, 3.8788167e-10, 5.5924204e-08, 7.3444756e-10]

    def test_train_in_cpu_infer_in_cpu(self):
       """
       cpu 下训练  cpu预测
       :return:
       """
       result = train_in_cpu_infer_in_cpu()
       cts_tools.check_data(result, self.cpu_infer, self.delta)

    def test_train_in_gpu_infer_in_cpu(self):
        """
        gpu 下训练  cpu预测
        :return:
        """
        result = train_in_gpu_infer_in_cpu()
        cts_tools.check_data(result, self.gpu_infer, self.delta)

    def test_train_in_cpu_infer_in_gpu(self):
        """
        cpu 下训练  gpu预测
        :return:
        """
        result = train_in_cpu_infer_in_gpu()
        cts_tools.check_data(result, self.cpu_infer, self.delta)

    def test_train_in_gpu_infer_in_gpu(self):
        """
        gpu 下训练  gpu预测
        :return:
        """
        result = train_in_gpu_infer_in_gpu()
        cts_tools.check_data(result, self.gpu_infer, self.delta)

    def test_save_in_cpu_load_in_cpu(self):
        """
        cpu save  cpu load
        :return:
        """
        result = save_in_cpu_load_in_cpu()
        cts_tools.check_data(result, self.cpu_infer, self.delta)

    def test_save_in_gpu_load_in_gpu(self):
        """
        gpu save  gpu load
        :return:
        """
        result = save_in_gpu_load_in_gpu()
        cts_tools.check_data(result, self.gpu_infer, self.delta)

    def test_save_in_cpu_load_in_cpu_with_lrdecay(self):
        """
        cpu train cpu load  with lr decay
        :return:
        """
        result = save_in_cpu_load_in_cpu_with_lrdecay()
        cts_tools.check_data(result, self.cpu_decay, 1e-2)

    def test_save_in_cpu_load_in_gpu_with_lrdecay(self):
        """
        cpu train gpu load  with lr decay
        :return:
        """
        result = save_in_cpu_load_in_gpu_with_lrdecay()
        cts_tools.check_data(result, self.cpu_decay, 1e-2)

    def test_save_in_gpu_load_in_cpu_with_lrdecay(self):
        """
        gpu train cpu load  with lr decay
        :return:
        """
        result = save_in_gpu_load_in_cpu_with_lrdecay()
        cts_tools.check_data(result, self.gpu_decay, self.delta)

    def test_save_in_gpu_load_in_gpu_with_lrdecay(self):
        """
        gpu train gpu load  with lr decay
        :return:
        """
        result = save_in_gpu_load_in_gpu_with_lrdecay()
        cts_tools.check_data(result, self.gpu_decay, self.delta)

    def test_load_wrongmodel_with_lrdecay(self):
        """
        test load wrong model
        :return:
        """
        load_wrongmodel_with_lrdecay()


if __name__ == "__main__":
    print(save_in_gpu_load_in_gpu_with_lrdecay())
