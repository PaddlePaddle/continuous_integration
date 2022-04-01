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
import time
import numpy as np
import paddle.inference as paddle_infer
import demo_helper as helper
import image_preprocess


def Inference(args, predictor) -> float:
    """
    paddle-inference
    Args:
        args : python input arguments
        predictor : paddle-inference predictor
    Returns:
        total_inference_cost (float): inference time
    """
    channels = int(args.image_shape.split(',')[0])
    height = int(args.image_shape.split(',')[1])
    width = int(args.image_shape.split(',')[2])

    im_size = height
    img_name = 'kite.jpg'
    img_path = os.path.join(args.model_name, img_name)
    img = np.array(cv2.imread(img_path))
    data = image_preprocess.preprocess(img, im_size)
    base_data = image_preprocess.preprocess(img, im_size)
    scale_factor = np.array([im_size * 1. / img.shape[0], im_size *
                            1. / img.shape[1]]).reshape((1, 2)).astype(np.float32)
    base_scale_factor = np.array([im_size * 1. / img.shape[0], im_size *
                                  1. / img.shape[1]]).reshape((1, 2)).astype(np.float32)
    im_shape = np.array([im_size, im_size]).reshape((1, 2)).astype(np.float32)
    base_im_shape = np.array([im_size, im_size]).reshape((1, 2)).astype(np.float32)
    for batch in range(args.batch_size - 1):
        data = np.concatenate((data, base_data), axis=0)
        scale_factor = np.concatenate((scale_factor, base_scale_factor), axis=0)
        im_shape = np.concatenate((im_shape, base_im_shape), axis=0)
    data_input = [im_shape, data, scale_factor]
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(data_input[i].shape)
        input_tensor.copy_from_cpu(data_input[i].copy())

    for i in range(args.warmup_times):
        predictor.run()
        output_names = predictor.get_output_names()
        output_hanlde = predictor.get_output_handle(output_names[0])
        output_data = output_hanlde.copy_to_cpu()

    time1 = time.time()
    for i in range(args.repeats):
        predictor.run()
        output_names = predictor.get_output_names()
        output_hanlde = predictor.get_output_handle(output_names[0])
        output_data = output_hanlde.copy_to_cpu()
    time2 = time.time()
    total_inference_cost = (time2 - time1) * 1000  # total latency, ms

    return total_inference_cost


def run_demo():
    """
    run_demo
    """
    args = helper.parse_args()
    config = helper.prepare_config(args)
    predictor_pool = paddle_infer.PredictorPool(config, 1)
    predictor = predictor_pool.retrive(0)
    total_time = Inference(args, predictor)

    helper.summary_config(config, args, total_time)


if __name__ == "__main__":
    run_demo()
