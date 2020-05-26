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
"""test static profiler."""

import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import numpy as np
import os


def test_profiler():
    """
    test profiler
    :return:
    """
    if os.path.exists("./profile"):
        os.remove("./profile")
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with profiler.profiler('CPU', 'total', './profile') as prof:
        with fluid.unique_name.guard():
            with fluid.program_guard(
                    main_program=main_program, startup_program=startup_program):
                epoc = 30
                dshape = [4, 3, 28, 28]
                data = fluid.layers.data(
                    name='data', shape=[3, 28, 28], dtype='float32')
                conv = fluid.layers.conv2d(
                    data, 20, 3, stride=[1, 1], padding=[1, 1])

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for i in range(epoc):
                    input = np.random.random(dshape).astype('float32')
                    exe.run(fluid.default_main_program(), feed={'data': input})
    if os.path.exists("./profile"):
        assert True
    else:
        assert False


def test_profiler1():
    """
    test profiler with sorted_key = 'total', 'calls', 'max', 'min', 'ave'
    :return:
    """
    sorted_key = 'calls'
    if os.path.exists("./profile"):
        os.remove("./profile")
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with profiler.profiler('CPU', sorted_key, './profile') as prof:
        with fluid.unique_name.guard():
            with fluid.program_guard(
                    main_program=main_program, startup_program=startup_program):
                epoc = 30
                dshape = [4, 3, 28, 28]
                data = fluid.layers.data(
                    name='data', shape=[3, 28, 28], dtype='float32')
                conv = fluid.layers.conv2d(
                    data, 20, 3, stride=[1, 1], padding=[1, 1])

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for i in range(epoc):
                    input = np.random.random(dshape).astype('float32')
                    exe.run(fluid.default_main_program(), feed={'data': input})
    if os.path.exists("./profile"):
        assert True
    else:
        assert False


def test_start_profiler():
    """
    test start_profiler
    :return:
    """
    if os.path.exists("./profile"):
        os.remove("./profile")
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            profiler.start_profiler('CPU')
            epoc = 30
            dshape = [4, 3, 28, 28]
            data = fluid.layers.data(
                name='data', shape=[3, 28, 28], dtype='float32')
            conv = fluid.layers.conv2d(
                data, 20, 3, stride=[1, 1], padding=[1, 1])

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(epoc):
                input = np.random.random(dshape).astype('float32')
                exe.run(fluid.default_main_program(), feed={'data': input})
            # for iter in range(10):
            #     if iter == 2:
            #         profiler.reset_profiler()
            # except each iteration
            profiler.stop_profiler('total', './profile')
    if os.path.exists("./profile"):
        assert True
    else:
        assert False


def test_start_profiler1():
    """
    test start_profiler state=GPU
    :return:
    """
    if os.path.exists("./profile"):
        os.remove("./profile")
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            profiler.start_profiler('GPU')
            epoc = 30
            dshape = [4, 3, 28, 28]
            data = fluid.layers.data(
                name='data', shape=[3, 28, 28], dtype='float32')
            conv = fluid.layers.conv2d(
                data, 20, 3, stride=[1, 1], padding=[1, 1])

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(epoc):
                input = np.random.random(dshape).astype('float32')
                exe.run(fluid.default_main_program(), feed={'data': input})
            # for iter in range(10):
            #     if iter == 2:
            #         profiler.reset_profiler()
            # except each iteration
            profiler.stop_profiler('total', './profile')
    if os.path.exists("./profile"):
        assert True
    else:
        assert False


def test_start_profiler2():
    """
    test start_profiler state=All
    :return:
    """
    if os.path.exists("./profile"):
        os.remove("./profile")
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            profiler.start_profiler('All')
            epoc = 30
            dshape = [4, 3, 28, 28]
            data = fluid.layers.data(
                name='data', shape=[3, 28, 28], dtype='float32')
            conv = fluid.layers.conv2d(
                data, 20, 3, stride=[1, 1], padding=[1, 1])

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(epoc):
                input = np.random.random(dshape).astype('float32')
                exe.run(fluid.default_main_program(), feed={'data': input})
            # for iter in range(10):
            #     if iter == 2:
            #         profiler.reset_profiler()
            # except each iteration
            profiler.stop_profiler('total', './profile')
    if os.path.exists("./profile"):
        assert True
    else:
        assert False


def test_start_profiler3():
    """
    test start_profiler state=nothing
    :return:
    """
    if os.path.exists("./profile"):
        os.remove("./profile")
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            try:
                profiler.start_profiler('nothing')
                epoc = 30
                dshape = [4, 3, 28, 28]
                data = fluid.layers.data(
                    name='data', shape=[3, 28, 28], dtype='float32')
                conv = fluid.layers.conv2d(
                    data, 20, 3, stride=[1, 1], padding=[1, 1])

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for i in range(epoc):
                    input = np.random.random(dshape).astype('float32')
                    exe.run(fluid.default_main_program(), feed={'data': input})
                # for iter in range(10):
                #     if iter == 2:
                #         profiler.reset_profiler()
                # except each iteration
                profiler.stop_profiler('total', './profile')
            except ValueError as e:
                print(e)
                assert True


def test_reset_profiler():
    """
    test reset profiler
    :return:
    """
    if os.path.exists("./profile"):
        os.remove("./profile")
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(
                main_program=main_program, startup_program=startup_program):
            profiler.start_profiler('All')
            epoc = 30
            dshape = [4, 3, 28, 28]
            data = fluid.layers.data(
                name='data', shape=[3, 28, 28], dtype='float32')
            conv = fluid.layers.conv2d(
                data, 20, 3, stride=[1, 1], padding=[1, 1])
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for i in range(epoc):
                input = np.random.random(dshape).astype('float32')
                exe.run(fluid.default_main_program(), feed={'data': input})
            for iter in range(10):
                if iter == 2:
                    profiler.reset_profiler()
            profiler.stop_profiler('total', './profile')
    if os.path.exists("./profile"):
        assert True
    else:
        assert False
