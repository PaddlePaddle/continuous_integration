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
"""test cuda profiler."""
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import numpy as np
import os


def test_cuda_profiler():
    """
    test cuda_profiler
    :return:
    """
    if fluid.is_compiled_with_cuda():
        if os.path.exists("./cuda_profile.txt"):
            os.remove("./cuda_profile.txt")
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with profiler.cuda_profiler("cuda_profile.txt", "csv") as prof:
            with fluid.unique_name.guard():
                with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                    epoc = 30
                    dshape = [4, 3, 28, 28]
                    data = fluid.layers.data(name='data', shape=[3, 28, 28], dtype='float32')
                    conv = fluid.layers.conv2d(data, 20, 3, stride=[1, 1], padding=[1, 1])

                    place = fluid.CUDAPlace(0)
                    exe = fluid.Executor(place)
                    exe.run(fluid.default_startup_program())
                    for i in range(epoc):
                        input = np.random.random(dshape).astype('float32')
                        exe.run(fluid.default_main_program(), feed={'data': input})
        if os.path.exists("./cuda_profile.txt"):
            assert True
        else:
            assert False


def test_cuda_profiler1():
    """
    test cuda_profiler with type = kvp
    :return:
    """
    if fluid.is_compiled_with_cuda():
        if os.path.exists("./cuda_profile.txt"):
            os.remove("./cuda_profile.txt")
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with profiler.cuda_profiler("cuda_profile.txt", "kvp") as prof:
            with fluid.unique_name.guard():
                with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                    epoc = 30
                    dshape = [4, 3, 28, 28]
                    data = fluid.layers.data(name='data', shape=[3, 28, 28], dtype='float32')
                    conv = fluid.layers.conv2d(data, 20, 3, stride=[1, 1], padding=[1, 1])

                    place = fluid.CUDAPlace(0)
                    exe = fluid.Executor(place)
                    exe.run(fluid.default_startup_program())
                    for i in range(epoc):
                        input = np.random.random(dshape).astype('float32')
                        exe.run(fluid.default_main_program(), feed={'data': input})
        if os.path.exists("./cuda_profile.txt"):
            assert True
        else:
            assert False

