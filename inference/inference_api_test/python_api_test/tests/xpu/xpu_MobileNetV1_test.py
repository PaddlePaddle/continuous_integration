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

import os
import cv2
import pytest
import numpy as np
import image_preprocess
from PIL import Image
from paddle.inference import Config
from paddle.inference import PrecisionType
from paddle.inference import create_predictor
from test_src import test_gpu_model_jetson


def inference_MobileNetV1(img, model_path, params_path):
    """
    inference_MobileNetV1
    Args:
        img: numpy img
        model_path: model path
        params_path: params path
    Returns:
        results : paddle inference output data
    """
    config = Config(model_path, params_path)
    config.enable_xpu(10 * 1024 * 1024)
    config.enable_lite_engine(PrecisionType.Float32, True) 
    config.switch_ir_optim(True)
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    config.enable_memory_optim()

    predictor = create_predictor(config)
    input_names = predictor.get_input_names()

    im_size = 224
    data = image_preprocess.preprocess(img, im_size)
    data_input = [data]
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(data_input[i].shape)
        input_tensor.copy_from_cpu(data_input[i].copy())

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)
    return results
    
@pytest.mark.p0
def test_MobileNetV1():
    """
    test_MobileNetV1
    Args:
        None
    Returns:
        None
    """
    diff_standard = 1e-5
    model_name = "MobileNetV1"
    test_model = test_gpu_model_jetson(model_name=model_name)
    model_path, params_path = test_model.test_comb_model_path("cv_class_model")
    img_name = 'bird.jpeg'
    image_path = test_model.test_readdata(
        path="cv_class_model", data_name=img_name)
    img = cv2.imread(image_path)
    with_lr_data = inference_MobileNetV1(img, model_path, params_path)
    npy_result = test_model.npy_result_path("cv_class_model")
    test_model.test_diff(npy_result, with_lr_data[0], diff_standard)
    
    # for test
    # np.save("MobileNetV1.npy",with_lr_data[0])
    # print(np.argmax(with_lr_data[0][0]))

