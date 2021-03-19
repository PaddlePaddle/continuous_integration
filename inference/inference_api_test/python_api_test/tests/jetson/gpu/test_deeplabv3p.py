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

import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor
import os
from test_src import test_gpu_model_jetson

def inference_deeplabv3p_resnet50(model_path,params_path,ir_optim=True):
    """
    inference_deeplabv3p
    Args:
        ir_optim(bool): with ir_optim or not
    Returns:
        output_data : paddle inference output_data
    """

    batch_size = 1
    config = Config(model_path, params_path)
    config.enable_use_gpu(1000, 0)
    config.switch_ir_optim(ir_optim)
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    config.enable_memory_optim()

    predictor = create_predictor(config)
    input_names = predictor.get_input_names()
    input_handle0 = predictor.get_input_handle(input_names[0])
    
    np.random.seed(0)
    fake_input0 = np.random.randn(batch_size, 3, 1024, 2048).astype("float32")
    input_handle0.reshape([1, 3, 1024, 2048])
    input_handle0.copy_from_cpu(fake_input0)
    
    predictor.run()
    
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()
    return output_data


def test_deeplabv3p_gpu():
    """
    test_deeplabv3p
    Args:
        None
    Returns:
        None
    """
    diff_standard = 1e-4
    model_name = "deeplabv3p_resnet50"
    test_model = test_gpu_model_jetson(model_name=model_name)
    model_path,params_path = test_model.test_comb_model_path("seg_inference_model")
    without_lr_data = inference_deeplabv3p_resnet50(model_path,params_path,ir_optim=False)
    with_lr_data = inference_deeplabv3p_resnet50(model_path,params_path,ir_optim=True)
    test_model.test_diff(without_lr_data,with_lr_data,diff_standard)