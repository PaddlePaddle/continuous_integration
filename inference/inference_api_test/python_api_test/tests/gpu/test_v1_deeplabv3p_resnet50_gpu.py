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
from paddle.inference import Config
from paddle.inference import create_predictor
from test_src import test_gpu_model_jetson

def inference_deeplabv3p_resnet50(img, model_path, params_path):
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
    config.switch_ir_optim(True)
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    config.enable_memory_optim()

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    data = image_preprocess.normalize(img,mean,std)
    data_input = np.array([data]).transpose([0, 3, 1, 2])

    predictor = create_predictor(config)
    input_names = predictor.get_input_names()
    input_handle0 = predictor.get_input_handle(input_names[0])
    input_handle0.reshape([batch_size, 3, 1024, 2048])
    input_handle0.copy_from_cpu(data_input)

    # do the inference
    predictor.run()

    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()

    return output_data

@pytest.mark.p0
@pytest.mark.p1
def test_deeplabv3p_resnet50():
    """
    test_deeplabv3p_resnet50
    Args:
        None
    Returns:
        None
    """
    diff_standard = 1e-4
    model_name = "deeplabv3p_resnet50"
    test_model = test_gpu_model_jetson(model_name=model_name)
    model_path, params_path = test_model.test_comb_model_path("cv_seg_model")
    img_name = 'seg_data.png'
    image_path = test_model.test_readdata(
        path="cv_seg_model", data_name=img_name)
    img = np.array(cv2.imread(image_path)).astype("float32")
    with_lr_data = inference_deeplabv3p_resnet50(img, model_path, params_path)
    npy_result = test_model.npy_result_path("cv_seg_model")
    test_model.test_diff(npy_result, with_lr_data[0], diff_standard)

    # save color img
    # np.save("deeplabv3p_resnet50.npy",with_lr_data[0])  #save npy
    # imgs = np.argmax(with_lr_data[0], axis=0)
    # color_img = image_preprocess.seg_color(np.array([imgs]).transpose([1,2,0]))
    # cv2.imwrite("deeplabv3p_resnet50.png", color_img)
