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
"""test static io."""
import paddle.fluid as fluid
import tools
import random
import numpy as np
import platform


def test_batch():
    """
    test batch
    :return:
    """

    def reader():
        """reader"""
        for i in range(10):
            yield i

    batch_reader = fluid.io.batch(reader, batch_size=2)
    res = list()
    for id, data in enumerate(batch_reader()):
        res.append(data)
    expect = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    tools.compare(res, expect)


def test_batch1():
    """
    test batch with batch_size = 1
    :return:
    """
    def reader():
        """reader"""
        for i in range(10):
            yield i

    batch_reader = fluid.io.batch(reader, batch_size=1)
    res = list()
    for id, data in enumerate(batch_reader()):
        res.append(data)
    expect = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    tools.compare(res, expect)


def test_batch2():
    """
    test batch with batch_size = 3 drop_last = False
    :return:
    """
    def reader():
        """reader"""
        for i in range(10):
            yield i

    batch_reader = fluid.io.batch(reader, batch_size=3)
    res = list()
    for id, data in enumerate(batch_reader()):
        res.append(data)
    expect = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert expect == res


def test_batch3():
    """
    test batch with batch_size = 3 drop_last = True
    :return:
    """
    def reader():
        """reader"""
        for i in range(10):
            yield i

    batch_reader = fluid.io.batch(reader, batch_size=3, drop_last=True)
    res = list()
    for id, data in enumerate(batch_reader()):
        res.append(data)
    expect = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    tools.compare(res, expect)


def test_buffered():
    """
    test buffered with size < 0
    :return:
    """
    def reader():
        """reader"""
        for i in range(10):
            yield i

    buffered_reader = fluid.io.buffered(reader, size=-100)
    batch_reader = fluid.io.batch(buffered_reader, batch_size=3, drop_last=True)
    res = list()
    for id, data in enumerate(batch_reader()):
        res.append(data)
    print(res)
    expect = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    tools.compare(res, expect)


def test_buffered1():
    """
    test buffered with size = 0
    :return:
    """
    def reader():
        """reader"""
        for i in range(10):
            yield i

    buffered_reader = fluid.io.buffered(reader, size=0)
    batch_reader = fluid.io.batch(buffered_reader, batch_size=3, drop_last=True)
    res = list()
    for id, data in enumerate(batch_reader()):
        res.append(data)
    print(res)
    expect = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    tools.compare(res, expect)


def test_buffered2():
    """
    test buffered with size > 0
    :return:
    """
    def reader():
        """
        reader
        :return:
        """
        for i in range(10):
            yield i

    buffered_reader = fluid.io.buffered(reader, size=100)
    batch_reader = fluid.io.batch(buffered_reader, batch_size=3, drop_last=True)
    res = list()
    for id, data in enumerate(batch_reader()):
        res.append(data)
    print(res)
    expect = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    tools.compare(res, expect)


def test_cache():
    """
    test cache
    :return:
    """
    def reader():
        """
        reader
        :return:
        """
        for i in range(10):
            yield i

    cache = fluid.io.cache(reader)
    batch_reader = fluid.io.batch(cache, batch_size=3, drop_last=True)
    res = list()
    for id, data in enumerate(batch_reader()):
        res.append(data)
    print(res)
    expect = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    tools.compare(res, expect)


def test_chain():
    """
    test chain  there are some bugs here
    :return:
    """

    def reader_creator_3(start):
        """
        reader
        :param start:
        :return:
        """
        def reader():
            """
            reader
            :return:
            """
            for i in range(start, start + 3):
                yield [i]

        return reader

    c = fluid.io.chain(reader_creator_3(0), reader_creator_3(10), reader_creator_3(20))
    res = list()
    for e in c():
        res.append(e)
    expect = [[0], [1], [2], [10], [11], [12], [20], [21], [22]]
    tools.compare(res, expect)


def test_compose():
    """
    test compose with check_alignment=True
    :return:
    """

    def reader1():
        """reader"""
        def reader():
            """reader"""
            for i in range(5):
                yield i

        return reader

    def reader2():
        """reader"""
        def reader():
            """reader"""
            for i in range(1, 5):
                yield i

        return reader

    def reader3():
        """reader"""
        def reader():
            """reader"""
            for i in range(2, 5):
                yield i

        return reader
    try:
        reader = fluid.io.compose(reader1(), reader2(), reader3(), check_alignment=True)
        res = list()
        for e in reader():
            res.append(e)
        expect = [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
        tools.compare(res, expect)
    except Exception as e:
        print(e)
        assert e.__str__() == "outputs of readers are not aligned."


def test_compose1():
    """
    test compose with check_alignment=False
    :return:
    """

    def reader1():
        """reader"""
        def reader():
            """reader"""
            for i in range(5):
                yield i

        return reader

    def reader2():
        """reader"""

        def reader():
            """reader"""
            for i in range(1, 5):
                yield i

        return reader

    def reader3():
        """reader"""

        def reader():
            """reader"""
            for i in range(2, 5):
                yield i

        return reader

    reader = fluid.io.compose(reader1(), reader2(), reader3(), check_alignment=False)
    res = list()
    for e in reader():
        res.append(e)
    expect = [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    tools.compare(res, expect)


def test_firstn():
    """
    test firstn
    :return:
    """

    def reader():
        """reader"""
        for i in range(100):
            yield i

    firstn_reader = fluid.io.firstn(reader, 5)
    res = list()
    for e in firstn_reader():
        res.append(e)
    expect = [0, 1, 2, 3, 4]
    tools.compare(res, expect)


def test_firstn1():
    """
    test firstn with n > len(range)
    :return:
    """
    def reader():
        """reader"""
        for i in range(10):
            yield i

    firstn_reader = fluid.io.firstn(reader, 12)
    res = list()
    for e in firstn_reader():
        res.append(e)
    expect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    tools.compare(res, expect)


def test_map_readers():
    """
    test map_readers
    :return:
    """
    def func(x, y):
        """
        mul
        :return:
        """
        return x * y

    def reader1():
        """reader"""
        for i in range(5):
            yield i

    def reader2():
        """reader"""
        for i in range(5):
            yield i

    reader = fluid.io.map_readers(func, reader1, reader2)
    res = list()
    for i in reader():
        res.append(i)

    expect = [0, 1, 4, 9, 16]
    tools.compare(res, expect)


def test_map_readers1():
    """
    test map_readers with no aligned readers
    :return:
    """
    def func(x, y):
        """
        mul
        :return:
        """
        return x * y

    def reader1():
        """reader"""
        for i in range(5):
            yield i

    try:
        reader = fluid.io.map_readers(func, reader1)
        res = list()
        for i in reader():
            res.append(i)

        expect = [0, 1, 4, 9, 16]
        tools.compare(res, expect)
    except TypeError as e:
        print(e)
        assert e.__str__() == "func() missing 1 required positional argument: 'y'"


def test_shuffle():
    """
    test shuffle
    :return:
    """
    random.seed(33)
    def reader():
        """
        reader
        :return:
        """
        for i in range(5):
            yield i

    shuffled_reader = fluid.io.shuffle(reader, 3)
    res = list()
    for e in shuffled_reader():
        res.append(e)
    expect = [1, 0, 2, 4, 3]
    tools.compare(res, expect)


def test_shuffle1():
    """
    test shuffle with buffer_size = 1
    :return:
    """
    random.seed(33)
    def reader():
        """
        reader
        :return:
        """
        for i in range(5):
            yield i

    shuffled_reader = fluid.io.shuffle(reader, 1)
    res = list()
    for e in shuffled_reader():
        res.append(e)
    expect = [0, 1, 2, 3, 4]
    tools.compare(res, expect)


def test_shuffle2():
    """
    test shuffle with buffer_size > reader length
    :return:
    """
    random.seed(33)
    def reader():
        """
        reader
        :return:
        """
        for i in range(5):
            yield i

    shuffled_reader = fluid.io.shuffle(reader, 1000)
    res = list()
    for e in shuffled_reader():
        res.append(e)
    print(res)
    expect = [3, 0, 2, 1, 4]
    tools.compare(res, expect)


def test_multiprocess_reader():
    """
    test multiprocess_reader
    :return:
    """

    def fake_reader(start, end):
        """reader"""
        def __impl__():
            """
            impl
            :return:
            """
            for i in range(start, end):
                yield [np.array([1]) * i],

        return __impl__

    if platform.system() == "Darwin" or platform.system() == "Linux":
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            place = fluid.CPUPlace()
            image = fluid.layers.data(
                name='image', dtype='int64', shape=[1])
            # fluid.layers.Print(image)
            reader = fluid.io.PyReader(
                feed_list=[image], capacity=2)
            image_p_1 = image + 1
            decorated_reader = fluid.io.multiprocess_reader(
                [fake_reader(1, 5), fake_reader(6, 10)], False)

            reader.decorate_sample_generator(decorated_reader, batch_size=2, places=[place])

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            res = list()
            for data in reader():
                res.append(exe.run(feed=data, fetch_list=[image])[0].tolist())
            # print(list(np.array(res).flat))
            res = list(np.array(res).flat)
            expect = [1, 2, 3, 4, 6, 7, 8, 9]

            tools.compare(res, expect)


def test_multiprocess_reader1():
    """
    test multiprocess_reader with use_pipe = True 有错误 http://newicafe.baidu.com:80/issue/DLTP-3237/show?from=page
    :return:
    """

    #



def test_multiprocess_reader2():
    """
    test multiprocess_reader with use_pipe = False quene_size = 2
    :return:
    """

    def fake_reader(start, end):
        """reader"""
        def __impl__():
            """
            impl
            :return:
            """
            for i in range(start, end):
                yield [np.array([1]) * i],

        return __impl__

    if platform.system() == "Darwin" or platform.system() == "Linux":
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            place = fluid.CPUPlace()
            image = fluid.layers.data(
                name='image', dtype='int64', shape=[1])
            # fluid.layers.Print(image)
            reader = fluid.io.PyReader(
                feed_list=[image], capacity=2)
            image_p_1 = image + 1
            decorated_reader = fluid.io.multiprocess_reader(
                [fake_reader(1, 5), fake_reader(6, 10)], False, queue_size=False)

            reader.decorate_sample_generator(decorated_reader, batch_size=2, places=[place])

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            res = list()
            for data in reader():
                res.append(exe.run(feed=data, fetch_list=[image])[0].tolist())
            # print(list(np.array(res).flat))
            res = list(np.array(res).flat)
            expect = [1, 2, 3, 4, 6, 7, 8, 9]
            tools.compare(res, expect)


def test_xmap_readers():
    """
    test xmap_readers
    :return:
    """

    def reader_creator():
        """reader"""
        def reader():
            """
            reader
            :return:
            """
            for i in range(10):
                yield i

        return reader

    def mapper(x):
        """mapper"""
        return (x + 1)

    thread_num = (1, 2, 4, 8, 16)
    buffer_size = (1, 2, 4, 8, 16)
    expect = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for t_num in thread_num:
        for size in buffer_size:
            user_reader = fluid.io.xmap_readers(mapper,
                                              reader_creator(),
                                              t_num, size, True)
            res = list()
            for i in user_reader():
                res.append(i)
            tools.compare(res, expect)


def test_xmap_readers1():
    """
    test xmap_readers with order = False
    :return:
    """

    def reader_creator():
        """reader"""
        def reader():
            """
            reader
            :return:
            """
            for i in range(10):
                yield i

        return reader

    def mapper(x):
        """mapper"""
        return (x + 1)

    thread_num = (1, 2, 4, 8, 16)
    buffer_size = (1, 2, 4, 8, 16)
    expect = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for t_num in thread_num:
        for size in buffer_size:
            user_reader = fluid.io.xmap_readers(mapper,
                                              reader_creator(),
                                              t_num, size, False)
            res = list()
            for i in user_reader():
                res.append(i)
            assert set(res) == set(expect)


def test_xmap_readers2():
    """
    test xmap_readers with order = False bufsize=1
    :return:
    """

    def reader_creator():
        """reader"""
        def reader():
            """
            reader
            :return:
            """
            for i in range(10):
                yield i

        return reader

    def mapper(x):
        """mapper"""
        return (x + 1)

    thread_num = (1, 2, 4, 8, 16)
    expect = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for t_num in thread_num:
        user_reader = fluid.io.xmap_readers(mapper,
                                          reader_creator(),
                                          t_num, 1, False)
        res = list()
        for i in user_reader():
            res.append(i)
        # print(res)
        assert set(res) == set(expect)


def test_xmap_readers3():
    """
    test xmap_readers with order = False thread=1
    :return:
    """

    def reader_creator():
        """reader"""
        def reader():
            """
            reader
            :return:
            """
            for i in range(10):
                yield i

        return reader

    def mapper(x):
        """mapper"""
        return (x + 1)

    buffer_size = (1, 2, 4, 8, 16)
    expect = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for size in buffer_size:
        user_reader = fluid.io.xmap_readers(mapper,
                                          reader_creator(),
                                          1, size, False)
        res = list()
        for i in user_reader():
            res.append(i)
        # print(res)
        tools.compare(res, expect)
        tools.compare(res, expect)


