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
"""test static  dataset."""

import paddle
import paddle.fluid as fluid
import re


def test_mnist_dataset():
    """
    test mnist dataset
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            train = paddle.batch(paddle.dataset.mnist.train(), batch_size=1, drop_last=True)
            length = len(list(enumerate(train())))
            assert length == 60000
            test = paddle.batch(paddle.dataset.mnist.test(), batch_size=1, drop_last=True)
            length = len(list(enumerate(test())))
            assert length == 10000


def test_cifar_dataset():
    """
    test cifar dataset
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            train = paddle.batch(paddle.dataset.cifar.train100(), batch_size=1, drop_last=True)
            length = len(list(enumerate(train())))
            assert length == 50000
            test = paddle.batch(paddle.dataset.cifar.test100(), batch_size=1, drop_last=True)
            length = len(list(enumerate(test())))
            assert length == 10000
            train = paddle.batch(paddle.dataset.cifar.train10(), batch_size=1, drop_last=True)
            length = len(list(enumerate(train())))
            assert length == 50000
            test = paddle.batch(paddle.dataset.cifar.test10(cycle=False), batch_size=1, drop_last=True)
            length = len(list(enumerate(test())))
            assert length == 10000
            test = paddle.batch(paddle.dataset.cifar.test10(cycle=True), batch_size=1, drop_last=True)
            length = 0
            for t in test():
                length = length + 1
                if length == 20000:
                    break
            assert length == 20000


def test_conll05_dataset():
    """
    test conll05 dataset
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            # test paddle.dataset.conll05.get_dict()
            assert type(paddle.dataset.conll05.get_dict()) == tuple
            # test paddle.dataset.conll05.get_embedding()
            assert type(paddle.dataset.conll05.get_embedding()) == str
            # test testdataset
            test = paddle.batch(paddle.dataset.conll05.test(), batch_size=1, drop_last=True)
            length = len(list(enumerate(test())))
            assert length == 5267


def test_imdb_dataset():
    """
    test imdb dataset
    :return:
    """
    TRAIN_PATTERN = re.compile("aclImdb/train/.*\.txt$")
    word_idx = paddle.dataset.imdb.build_dict(TRAIN_PATTERN, 150)
    assert len(word_idx) == 7036
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            train = paddle.dataset.imdb.train(word_idx)
            length = len(list(enumerate(train())))
            assert length == 25000
            test = paddle.dataset.imdb.test(word_idx)
            length = len(list(enumerate(test())))
            assert length == 25000


def test_imikolov_dataset():
    """
    test imikolov dataset
    :return:
    """
    dict = paddle.dataset.imikolov.build_dict(min_word_freq=50)
    assert len(paddle.dataset.imikolov.build_dict(min_word_freq=50)) == 2074
    # test different n
    train = paddle.dataset.imikolov.train(dict, 10, data_type=1)
    assert len(list(enumerate(train()))) == 599453
    train = paddle.dataset.imikolov.train(dict, 20, data_type=1)
    assert len(list(enumerate(train()))) == 263169
    # test different data_type
    train = paddle.dataset.imikolov.train(dict, 10, data_type=2)
    assert len(list(enumerate(train()))) == 4956
    train = paddle.dataset.imikolov.train(dict, 20, data_type=2)
    assert len(list(enumerate(train()))) == 19867


def test_movielens_dataset():
    """
    test movielens dataset
    :return:
    """
    assert len(paddle.dataset.movielens.get_movie_title_dict()) == 5174
    assert paddle.dataset.movielens.max_movie_id() == 3952
    assert paddle.dataset.movielens.max_user_id() == 6040
    assert paddle.dataset.movielens.max_job_id() == 20
    assert len(paddle.dataset.movielens.movie_categories()) == 18
    assert len(paddle.dataset.movielens.user_info()) == 6040
    assert len(paddle.dataset.movielens.movie_info()) == 3883


def test_sentiment_dataset():
    """
    test sentiment dataset
    :return:
    """
    word_dict = len(paddle.dataset.sentiment.get_word_dict())
    assert word_dict == 39768
    train = paddle.dataset.sentiment.train()
    assert len(list(enumerate(train))) == 1600
    test = paddle.dataset.sentiment.test()
    assert len(list(enumerate(test))) == 400


def test_uci_housing_dataset():
    """
    test uci_housing dataset
    :return:
    """
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(train_program, startup_program):
            train = paddle.batch(paddle.dataset.uci_housing.train(), batch_size=1, drop_last=True)
            length = len(list(enumerate(train())))
            assert length == 404
            test = paddle.batch(paddle.dataset.uci_housing.test(), batch_size=1, drop_last=True)
            length = len(list(enumerate(test())))
            assert length == 102


def test_wmt14_dataset():
    """
    test wmt14 dataset
    :return:
    """
    train = paddle.dataset.wmt14.train(100)
    assert len(list(enumerate(train()))) == 191155
    # check different dict_size
    train = paddle.dataset.wmt14.train(5)
    assert len(list(enumerate(train()))) == 191155
    test = paddle.dataset.wmt14.test(100)
    assert len(list(enumerate(test()))) == 5957
    # check different dict_size
    test = paddle.dataset.wmt14.test(5)
    assert len(list(enumerate(test()))) == 5957


def test_wmt16_dataset():
    """
    test wmt16 dataset
    :return:
    """
    train = paddle.dataset.wmt16.train(50, 50, src_lang='en')
    assert len(list(enumerate(train()))) == 29000
    train = paddle.dataset.wmt16.train(500, 250, src_lang='en')
    assert len(list(enumerate(train()))) == 29000
    test = paddle.dataset.wmt16.test(50, 50, src_lang='en')
    assert len(list(enumerate(test()))) == 1000
    test = paddle.dataset.wmt16.test(500, 250, src_lang='en')
    assert len(list(enumerate(test()))) == 1000
