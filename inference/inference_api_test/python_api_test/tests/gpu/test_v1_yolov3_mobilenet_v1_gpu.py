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
from paddle.inference import create_predictor
from test_src import test_gpu_model_jetson


def inference_yolov3_mobilenet_v1(img, model_path, params_path, with_ir=True):
    """
    inference_ttfnet
    Args:
        img: numpy img
        model_path: model path
        params_path: params path
    Returns:
        results : paddle inference output data
    """
    batch_size = 1
    config = Config(model_path, params_path)
    config.enable_use_gpu(0)
    config.switch_ir_optim(with_ir)
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    config.enable_memory_optim()

    predictor = create_predictor(config)
    input_names = predictor.get_input_names()

    im_size = 608
    data = image_preprocess.preprocess(img, im_size)
    scale_factor = np.array([im_size * 1. / img.shape[0], im_size *
                             1. / img.shape[1]]).reshape((1, 2)).astype(np.float32)
    im_shape = np.array([im_size, im_size]).reshape((1, 2)).astype(np.float32)
    data_input = [im_shape, data, scale_factor]

    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(data_input[i].shape)
        input_tensor.copy_from_cpu(data_input[i])

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
def test_yolov3_mobilenet_v1():
    """
    test_yolov3_mobilenet_v1
    Args:
        None
    Returns:
        None
    """
    diff_standard = 1e-4
    model_name = "yolov3_mobilenet_v1"
    test_model = test_gpu_model_jetson(model_name=model_name)
    model_path, params_path = test_model.test_comb_model_path(
        "cv_detect_model")
    # with_lr_data = inference_yolov3_mobilenet_v1(model_path,params_path,ir_optim=True)
    img_name = 'kite.jpg'
    image_path = test_model.test_readdata(
        path="cv_detect_model", data_name=img_name)
    img = cv2.imread(image_path)
    with_ir_data = inference_yolov3_mobilenet_v1(img, model_path, params_path)
    no_ir_data = inference_yolov3_mobilenet_v1(img, model_path, params_path, with_ir=False)

    # npy_result = test_model.npy_result_path("cv_detect_model")
    test_model.test_diff(no_ir_data[0], with_ir_data[0], diff_standard)

    # det image with box
    # np.save("yolov3_mobilenet_v1.npy",with_lr_data[0])
    # image_preprocess.draw_bbox(
    # image_path, with_lr_data[0], save_name="yolov3_mobilenet_v1.jpg")