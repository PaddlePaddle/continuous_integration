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

import argparse
import time
import logging

import numpy as np
import paddle.fluid.inference as paddle_infer

import demo_helper as helper


def Inference(args, predictor) -> int:
    """
    paddle-inference
    Args:
        args : python input arguments
        predictor : paddle-inference predictor
    Returns:
        total_inference_cost (int): inference time
    """
    batch_size = args.batch_size
    seq_length = 128

    # src_ids, int64[?,?,1]
    # pos_ids, int64[?,?,1]
    # sent_ids, int64[?,?,1]
    # input mask, float32[?,?,1]
    # input_num = batch_size * seq_length * 1

    # prepare inputs
    src_ids_data = np.ones((args.batch_size, 128, 1)).astype(np.int64)
    pos_ids_data = np.ones((args.batch_size, 128, 1)).astype(np.int64)
    sent_ids_data = np.ones((args.batch_size, 128, 1)).astype(np.int64)
    mask_data = np.ones((args.batch_size, 128, 1)).astype(np.float32)

    in_names = predictor.get_input_names()
    # set inputs
    src_ids = predictor.get_input_handle(in_names[0])
    src_ids.reshape([args.batch_size, seq_length, 1])
    src_ids.copy_from_cpu(src_ids_data)

    pos_ids = predictor.get_input_handle(in_names[1])
    pos_ids.reshape([args.batch_size, seq_length, 1])
    pos_ids.copy_from_cpu(pos_ids_data)

    sent_ids = predictor.get_input_handle(in_names[2])
    sent_ids.reshape([args.batch_size, seq_length, 1])
    sent_ids.copy_from_cpu(sent_ids_data)

    mask= predictor.get_input_handle(in_names[3])
    mask.reshape([args.batch_size, seq_length, 1])
    mask.copy_from_cpu(mask_data)

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

def dynamic_shsape_config(config):
    """
    set dynamic shape config for bert
    Args:
        config : paddle-inference config
    Returns:
        config : paddle-inference config
    """
    names = ["src_ids", "pos_ids", "sent_ids", "input_mask"]
    # head_number = 12
    min_input_shape = [1, 1, 1]
    max_input_shape = [100, 128, 1]
    opt_input_shape = [10, 60, 1]

    config.set_trt_dynamic_shape_info(
      {names[0]:min_input_shape, names[1]:min_input_shape, names[2]:min_input_shape, names[3] : [1, 1, 1]},
      {names[0]:max_input_shape, names[1]:max_input_shape, names[2]:max_input_shape, names[3] : [100, 128, 128]},
      {names[0]:opt_input_shape, names[1]:opt_input_shape, names[2]:opt_input_shape, names[3] : [10, 60, 60]})

    return config


def run_demo():
    """
    run_demo
    """
    args = helper.parse_args()
    config = helper.prepare_config(args)
    if args.use_trt:
        config = dynamic_shsape_config(config)
    predictor_pool = paddle_infer.PredictorPool(config, 1)
    predictor = predictor_pool.retrive(0)
    total_time = Inference(args, predictor)

    helper.summary_config(config, args, total_time)

if __name__ == "__main__":
    run_demo()
