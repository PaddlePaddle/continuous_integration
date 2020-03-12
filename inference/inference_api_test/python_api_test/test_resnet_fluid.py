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
""" Cases for inference, ResNet50Model. """

import math
import yaml
import platform
import nose.tools as tools
import api_infer

PYTHON_VERSION = platform.python_version()
CASE_ROOT = "Test_Case/ResNet50"
FILE_ROOT = "Data/python/resnet50/test_image.jpg"
MODEL_ROOT = "Data/python/resnet50/model"


def parse_case(case_file):
    """
    parse cases from config file
    Args:
        case_file(string|.yaml): case directory
    Returns:
        config: a dict that contains configuration
    """
    yaml_config = open(case_file)
    config = yaml.load(yaml_config)
    config['model_dir'] = MODEL_ROOT
    return config


def retrive_data(model, data_style="fake"):
    """
    get input data

    Args:
        model(class): Instance class of model
        data_style(string): input data style, three options: fake, random, real

    Returns:
        input_data: list of data
    """
    if data_style == "fake":
        input_data = model.load_fake_data(1, 3, 224, 224)
    elif data_style == "random":
        input_data = model.load_random_data(1, 3, 224, 224)
    elif data_style == "real":
        input_data = model.load_real_data(FILE_ROOT, 1, 3, 224, 224)
    else:
        raise ValueError(
            "parameter data_style should be 'fake', 'random' or 'real' ")

    return input_data


def run_infer(model, case_file, input_data):
    """
    run inference

    Args:
        model(class): Instance class of model
        case_file(string): case directory
        input_data(numpy array): input_data for inference

    Returns:
        res: inference_result
    """
    config = parse_case(case_file)
    model.set_config(config)
    res = model.run(input_data)

    return res


def check_float(a, b, precision=1e-4):
    """
    check float data
    Args:
        a(list): input_data1
        b(list): input_data2
        precision(float): precision for checking diff for a and b
    Returns:
        bool
    """

    def __adjust_data(num):
        if num == 0.0:
            return 0.0
        return num / 10**(math.floor(math.log10(abs(num))) + 1)

    a = __adjust_data(a)
    b = __adjust_data(b)
    return math.fabs(a - b) < precision


class TestResNet50API(object):
    """
    ResNet50 Model Test

    Attributes:
        None
    """

    def __init__(self):
        self.model = api_infer.ResNet50Model()
        self.input_data = retrive_data(self.model, data_style="real")

    def test_cpu_gpu_result(self, precision=1e-1):
        """
        test cpu and gpu infer data

        Args:
            precision (float): The precision for checking. 


        """
        res1 = run_infer(self.model, CASE_ROOT + "/resnet_fluid_gpu.yaml",
                         self.input_data)
        res2 = run_infer(self.model, CASE_ROOT + "/resnet_fluid_cpu.yaml",
                         self.input_data)
        result1 = res1[0].data.float_data()
        result2 = res2[0].data.float_data()
        for i in range(len(result1)):
            tools.assert_almost_equal(result1[i], result2[i], delta=precision)

    def test_cpu_mkldnn_result(self, precision=1e-3):
        """
        test cpu and cpu mkldnn infer data

        Args:
            precision (float): The precision for checking. 
        """
        res1 = run_infer(self.model, CASE_ROOT + "/resnet_fluid_cpu.yaml",
                         self.input_data)
        res2 = run_infer(self.model, CASE_ROOT + "/resnet_mkldnn_cpu.yaml",
                         self.input_data)
        result1 = res1[0].data.float_data()
        result2 = res2[0].data.float_data()
        for i in range(len(result1)):
            tools.assert_almost_equal(result1[i], result2[i], delta=precision)
