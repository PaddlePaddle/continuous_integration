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
"""test dataloader."""
import paddle.fluid as fluid
import paddle
import time
import numpy as np


def reader_decorator(reader):
    """
    reader_decorator
    :param reader:
    :return:
    """
    def __reader__():
        """
        __reader__
        :return:
        """
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(1, 28, 28)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield img, label
    return __reader__


def test_single_process_data_loader():
    """
    test single process data loader
    :return:
    """
    with fluid.dygraph.guard():
        train_reader = paddle.batch(
                    reader_decorator(
                        paddle.dataset.mnist.train()),
                        batch_size=1,
                        drop_last=True)
        train_loader = fluid.io.DataLoader.from_generator(capacity=10)
        train_loader.set_sample_list_generator(train_reader, places=fluid.CPUPlace())
        stime = time.time()
        a = list(train_reader())
        time1 = time.time()
        b = list(train_loader())
        time2 = time.time()
        print(time1-stime)
        print(time2-time1)
        assert len(a) == len(b)
        for i in range(len(a)):
            if a[i][0][1][0] != b[i][1].numpy()[0][0]:
                assert False


def next_sample(reader):
    """
    next
    :param reader:
    :return:
    """
    try:
        sample = next(reader)
    except StopIteration:
        sample = None
    return sample


def test_multi_process_data_loader():
    """
    test multi process data loader
    :return:
    """
    with fluid.dygraph.guard():
        train_reader = paddle.batch(
                    reader_decorator(
                        paddle.dataset.mnist.train()),
                        batch_size=1,
                        drop_last=True)
        train_loader = fluid.io.DataLoader.from_generator(capacity=2, use_multiprocess=True)
        train_loader.set_sample_list_generator(train_reader, places=fluid.CPUPlace())
        a = list(train_reader())
        train_loader_iter = train_loader.__iter__()
        for i in range(len(a)):
            b = next_sample(train_loader_iter)
            if b is None:
                assert False
            if a[i][0][1][0] != b[1].numpy()[0][0]:
                assert False