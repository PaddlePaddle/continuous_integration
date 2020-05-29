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
"""test compatible saveload."""
from __future__ import print_function

import numpy
import paddle
import paddle.fluid as fluid
import shutil

BATCH_SIZE = 64
PASS_NUM = 1
use_cuda = True if fluid.core.is_compiled_with_cuda() else False
predict = 'convolutional_neural_network'


def loss_net(hidden, label):
    """
    loss net
    :param hidden:
    :param label:
    :return:
    """
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc


def multilayer_perceptron(img, label):
    """
    multilayer_perceptron
    :param img:
    :param label:
    :return:
    """
    img = fluid.layers.fc(input=img, size=200, act='tanh')
    hidden = fluid.layers.fc(input=img, size=200, act='tanh')
    return loss_net(hidden, label)


def softmax_regression(img, label):
    """
    softmax
    :param img:
    :param label:
    :return:
    """
    return loss_net(img, label)


def convolutional_neural_network(img, label):
    """
    cnn
    :param img:
    :param label:
    :return:
    """
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    return loss_net(conv_pool_2, label)


def train1(nn_type,
           use_cuda,
           save_dirname=None,
           model_filename=None,
           params_filename=None):
    """
    train
    :param nn_type:
    :param use_cuda:
    :param save_dirname:
    :param model_filename:
    :param params_filename:
    :return:
    """
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    startup_program = fluid.Program()
    main_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        with fluid.unique_name.guard():
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=BATCH_SIZE)
            test_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
            startup_program.random_seed = 90
            main_program.random_seed = 90

            img = fluid.data(
                name='img', shape=[None, 1, 28, 28], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')

            if nn_type == 'softmax_regression':
                net_conf = softmax_regression
            elif nn_type == 'multilayer_perceptron':
                net_conf = multilayer_perceptron
            else:
                net_conf = convolutional_neural_network

            prediction, avg_loss, acc = net_conf(img, label)
            test_program = main_program.clone(for_test=True)
            test_program1 = main_program.clone(for_test=True)
            optimizer = fluid.optimizer.Adam(learning_rate=0.001)
            optimizer.minimize(avg_loss)

            def load(train_test_program, train_test_feed, train_test_reader):
                """
                test new load api
                :param train_test_program:
                :param train_test_feed:
                :param train_test_reader:
                :return:
                """
                acc_set = []
                avg_loss_set = []
                param_path = "./compatible_save_param"
                fluid.load(train_test_program, param_path, exe)
                for test_data in train_test_reader():
                    acc_np, avg_loss_np = exe.run(
                        program=train_test_program,
                        feed=train_test_feed.feed(test_data),
                        fetch_list=[acc, avg_loss])
                    acc_set.append(float(acc_np))
                    avg_loss_set.append(float(avg_loss_np))
                # get test acc and loss
                acc_val_mean = numpy.array(acc_set).mean()
                avg_loss_val_mean = numpy.array(avg_loss_set).mean()
                return avg_loss_val_mean, acc_val_mean

            def train_test(train_test_program, train_test_feed,
                           train_test_reader):
                """
                test
                :param train_test_program:
                :param train_test_feed:
                :param train_test_reader:
                :return:
                """
                acc_set = []
                avg_loss_set = []
                param_path = "./compatible_save_param"
                fluid.io.load_params(
                    executor=exe,
                    dirname=param_path,
                    main_program=train_test_program)
                for test_data in train_test_reader():
                    acc_np, avg_loss_np = exe.run(
                        program=train_test_program,
                        feed=train_test_feed.feed(test_data),
                        fetch_list=[acc, avg_loss])
                    acc_set.append(float(acc_np))
                    avg_loss_set.append(float(avg_loss_np))
                # get test acc and loss
                acc_val_mean = numpy.array(acc_set).mean()
                avg_loss_val_mean = numpy.array(avg_loss_set).mean()
                return avg_loss_val_mean, acc_val_mean

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

            exe = fluid.Executor(place)

            feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
            exe.run(startup_program)
            epochs = [epoch_id for epoch_id in range(PASS_NUM)]

            lists = []
            step = 0
            for epoch_id in epochs:
                for step_id, data in enumerate(train_reader()):
                    metrics = exe.run(main_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_loss, acc])
                    if step % 100 == 0:
                        print("Pass %d, Epoch %d, Cost %f" % (step, epoch_id,
                                                              metrics[0]))
                    step += 1
                if save_dirname is not None:
                    fluid.io.save_params(exe, "./compatible_save_param",
                                         main_program)
                    # load_param(test_program1)
                    # test for epoch
                    avg_loss_val, acc_val = train_test(
                        train_test_program=test_program,
                        train_test_reader=test_reader,
                        train_test_feed=feeder)
                    avg_loss_val1, acc_val1 = load(
                        train_test_program=test_program1,
                        train_test_reader=test_reader,
                        train_test_feed=feeder)
                    print("Test with Epoch %d, avg_cost: %s, acc: %s" %
                          (epoch_id, avg_loss_val, acc_val))
                    print("New Test with Epoch %d, avg_cost: %s, acc: %s" %
                          (epoch_id, avg_loss_val1, acc_val1))
                    assert avg_loss_val == avg_loss_val1
                    assert acc_val == acc_val1
                    lists.append((epoch_id, avg_loss_val, acc_val))
                    shutil.rmtree("./compatible_save_param")

            # find the best pass
            best = sorted(lists, key=lambda list: float(list[1]))[0]
            print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
            print('The classification accuracy is %.2f%%' %
                  (float(best[2]) * 100))


def train2(nn_type,
           use_cuda,
           save_dirname=None,
           model_filename=None,
           params_filename=None):
    """
    train
    :param nn_type:
    :param use_cuda:
    :param save_dirname:
    :param model_filename:
    :param params_filename:
    :return:
    """
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    startup_program = fluid.Program()
    main_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        with fluid.unique_name.guard():
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=BATCH_SIZE)
            test_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
            startup_program.random_seed = 90
            main_program.random_seed = 90

            img = fluid.data(
                name='img', shape=[None, 1, 28, 28], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')

            if nn_type == 'softmax_regression':
                net_conf = softmax_regression
            elif nn_type == 'multilayer_perceptron':
                net_conf = multilayer_perceptron
            else:
                net_conf = convolutional_neural_network

            prediction, avg_loss, acc = net_conf(img, label)
            test_program = main_program.clone(for_test=True)
            test_program1 = main_program.clone(for_test=True)
            optimizer = fluid.optimizer.Adam(learning_rate=0.001)
            optimizer.minimize(avg_loss)

            def load(train_test_program, train_test_feed, train_test_reader):
                """
                test new load api
                :param train_test_program:
                :param train_test_feed:
                :param train_test_reader:
                :return:
                """
                acc_set = []
                avg_loss_set = []
                param_path = "./compatible_save_persist"
                fluid.load(train_test_program, param_path, exe)
                for test_data in train_test_reader():
                    acc_np, avg_loss_np = exe.run(
                        program=train_test_program,
                        feed=train_test_feed.feed(test_data),
                        fetch_list=[acc, avg_loss])
                    acc_set.append(float(acc_np))
                    avg_loss_set.append(float(avg_loss_np))
                # get test acc and loss
                acc_val_mean = numpy.array(acc_set).mean()
                avg_loss_val_mean = numpy.array(avg_loss_set).mean()
                return avg_loss_val_mean, acc_val_mean

            def train_test(train_test_program, train_test_feed,
                           train_test_reader):
                """
                test
                :param train_test_program:
                :param train_test_feed:
                :param train_test_reader:
                :return:
                """
                acc_set = []
                avg_loss_set = []
                param_path = "./compatible_save_persist"
                fluid.io.load_persistables(
                    executor=exe,
                    dirname=param_path,
                    main_program=train_test_program)
                for test_data in train_test_reader():
                    acc_np, avg_loss_np = exe.run(
                        program=train_test_program,
                        feed=train_test_feed.feed(test_data),
                        fetch_list=[acc, avg_loss])
                    acc_set.append(float(acc_np))
                    avg_loss_set.append(float(avg_loss_np))
                # get test acc and loss
                acc_val_mean = numpy.array(acc_set).mean()
                avg_loss_val_mean = numpy.array(avg_loss_set).mean()
                return avg_loss_val_mean, acc_val_mean

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

            exe = fluid.Executor(place)

            feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
            exe.run(startup_program)
            epochs = [epoch_id for epoch_id in range(PASS_NUM)]

            lists = []
            step = 0
            for epoch_id in epochs:
                for step_id, data in enumerate(train_reader()):
                    metrics = exe.run(main_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_loss, acc])
                    if step % 100 == 0:
                        print("Pass %d, Epoch %d, Cost %f" % (step, epoch_id,
                                                              metrics[0]))
                    step += 1
                if save_dirname is not None:
                    fluid.io.save_persistables(exe, "./compatible_save_persist",
                                               main_program)
                    # load_param(test_program1)
                    # test for epoch
                    avg_loss_val, acc_val = train_test(
                        train_test_program=test_program,
                        train_test_reader=test_reader,
                        train_test_feed=feeder)
                    avg_loss_val1, acc_val1 = load(
                        train_test_program=test_program1,
                        train_test_reader=test_reader,
                        train_test_feed=feeder)
                    print("Test with Epoch %d, avg_cost: %s, acc: %s" %
                          (epoch_id, avg_loss_val, acc_val))
                    print("New Test with Epoch %d, avg_cost: %s, acc: %s" %
                          (epoch_id, avg_loss_val1, acc_val1))
                    assert avg_loss_val == avg_loss_val1
                    assert acc_val == acc_val1
                    lists.append((epoch_id, avg_loss_val, acc_val))
                    shutil.rmtree("./compatible_save_persist")

            # find the best pass
            best = sorted(lists, key=lambda list: float(list[1]))[0]
            print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
            print('The classification accuracy is %.2f%%' %
                  (float(best[2]) * 100))


def train3(nn_type,
           use_cuda,
           save_dirname=None,
           model_filename=None,
           params_filename=None):
    """
    train
    :param nn_type:
    :param use_cuda:
    :param save_dirname:
    :param model_filename:
    :param params_filename:
    :return:
    """
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    startup_program = fluid.Program()
    main_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        with fluid.unique_name.guard():
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=BATCH_SIZE)
            test_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
            startup_program.random_seed = 90
            main_program.random_seed = 90

            img = fluid.data(
                name='img', shape=[None, 1, 28, 28], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')

            if nn_type == 'softmax_regression':
                net_conf = softmax_regression
            elif nn_type == 'multilayer_perceptron':
                net_conf = multilayer_perceptron
            else:
                net_conf = convolutional_neural_network

            prediction, avg_loss, acc = net_conf(img, label)
            test_program = main_program.clone(for_test=True)
            test_program1 = main_program.clone(for_test=True)
            optimizer = fluid.optimizer.Adam(learning_rate=0.001)
            optimizer.minimize(avg_loss)

            def load(train_test_program, train_test_feed, train_test_reader,
                     vars):
                """
                test new load api
                :param train_test_program:
                :param train_test_feed:
                :param train_test_reader:
                :return:
                """
                acc_set = []
                avg_loss_set = []
                param_path = "./compatible_save_vars"
                fluid.load(train_test_program, param_path, exe)
                for test_data in train_test_reader():
                    acc_np, avg_loss_np = exe.run(
                        program=train_test_program,
                        feed=train_test_feed.feed(test_data),
                        fetch_list=[acc, avg_loss])
                    acc_set.append(float(acc_np))
                    avg_loss_set.append(float(avg_loss_np))
                # get test acc and loss
                acc_val_mean = numpy.array(acc_set).mean()
                avg_loss_val_mean = numpy.array(avg_loss_set).mean()
                return avg_loss_val_mean, acc_val_mean

            def train_test(train_test_program, train_test_feed,
                           train_test_reader, vars):
                """
                test
                :param train_test_program:
                :param train_test_feed:
                :param train_test_reader:
                :return:
                """
                acc_set = []
                avg_loss_set = []
                param_path = "./compatible_save_vars"
                fluid.io.load_vars(
                    executor=exe,
                    dirname=param_path,
                    main_program=train_test_program,
                    vars=vars)
                for test_data in train_test_reader():
                    acc_np, avg_loss_np = exe.run(
                        program=train_test_program,
                        feed=train_test_feed.feed(test_data),
                        fetch_list=[acc, avg_loss])
                    acc_set.append(float(acc_np))
                    avg_loss_set.append(float(avg_loss_np))
                # get test acc and loss
                acc_val_mean = numpy.array(acc_set).mean()
                avg_loss_val_mean = numpy.array(avg_loss_set).mean()
                return avg_loss_val_mean, acc_val_mean

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

            exe = fluid.Executor(place)

            feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
            exe.run(startup_program)
            epochs = [epoch_id for epoch_id in range(PASS_NUM)]

            lists = []
            step = 0
            for epoch_id in epochs:
                for step_id, data in enumerate(train_reader()):
                    metrics = exe.run(main_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_loss, acc])
                    if step % 100 == 0:
                        print("Pass %d, Epoch %d, Cost %f" % (step, epoch_id,
                                                              metrics[0]))
                    step += 1
                if save_dirname is not None:
                    vars = fluid.io.get_program_parameter(main_program)
                    fluid.io.save_vars(
                        exe, "./compatible_save_vars", main_program, vars=vars)
                    # load_param(test_program1)
                    # test for epoch
                    avg_loss_val, acc_val = train_test(
                        train_test_program=test_program,
                        train_test_reader=test_reader,
                        train_test_feed=feeder,
                        vars=vars)
                    avg_loss_val1, acc_val1 = load(
                        train_test_program=test_program1,
                        train_test_reader=test_reader,
                        train_test_feed=feeder,
                        vars=vars)
                    print("Test with Epoch %d, avg_cost: %s, acc: %s" %
                          (epoch_id, avg_loss_val, acc_val))
                    print("New Test with Epoch %d, avg_cost: %s, acc: %s" %
                          (epoch_id, avg_loss_val1, acc_val1))
                    assert avg_loss_val == avg_loss_val1
                    assert acc_val == acc_val1
                    lists.append((epoch_id, avg_loss_val, acc_val))
                    shutil.rmtree("./compatible_save_vars")

            # find the best pass
            best = sorted(lists, key=lambda list: float(list[1]))[0]
            print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
            print('The classification accuracy is %.2f%%' %
                  (float(best[2]) * 100))


def main1(use_cuda, nn_type):
    """
    main
    :param use_cuda:
    :param nn_type:
    :return:
    """
    model_filename = None
    params_filename = None
    save_dirname = "recognize_digits_" + nn_type + ".inference.model"

    # call train() with is_local argument to run distributed train
    train1(
        nn_type=nn_type,
        use_cuda=use_cuda,
        save_dirname=save_dirname,
        model_filename=model_filename,
        params_filename=params_filename)


def main2(use_cuda, nn_type):
    """
    main
    :param use_cuda:
    :param nn_type:
    :return:
    """
    model_filename = None
    params_filename = None
    save_dirname = "recognize_digits_" + nn_type + ".inference.model"

    # call train() with is_local argument to run distributed train
    train2(
        nn_type=nn_type,
        use_cuda=use_cuda,
        save_dirname=save_dirname,
        model_filename=model_filename,
        params_filename=params_filename)


def main3(use_cuda, nn_type):
    """
    main
    :param use_cuda:
    :param nn_type:
    :return:
    """
    model_filename = None
    params_filename = None
    save_dirname = "recognize_digits_" + nn_type + ".inference.model"

    # call train() with is_local argument to run distributed train
    train3(
        nn_type=nn_type,
        use_cuda=use_cuda,
        save_dirname=save_dirname,
        model_filename=model_filename,
        params_filename=params_filename)


def test_param():
    """
    start test save param
    :return:
    """
    main1(use_cuda=use_cuda, nn_type=predict)


def test_persist():
    """
    start test save persist
    :return:
    """
    main2(use_cuda=use_cuda, nn_type=predict)


def test_vars():
    """
    start test save vars
    :return:
    """
    main3(use_cuda=use_cuda, nn_type=predict)
