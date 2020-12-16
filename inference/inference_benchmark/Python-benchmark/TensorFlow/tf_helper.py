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
import tensorflow.compat.v1 as tf

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--model_type", type=str, default="static",
                        help="model generate type")
    parser.add_argument("--model_path", type=str, help="model filename")
    parser.add_argument("--params_path", type=str, default="",
                        help="parameter filename")
    parser.add_argument("--trt_precision", type=str, default="fp32",
                        help="trt precision, choice = ['fp32', 'fp16', 'int8']")
    parser.add_argument("--image_shape", type=str, default="3,224,224",
                        help="can only use for one input model(e.g. image classification)")
    parser.add_argument("--input_node", type=str, default="inputs:0",
                        help="tf model input node")
    parser.add_argument("--output_node", type=str, default="predictions/Reshape_1:0",
                        help="tf model output node")

    parser.add_argument("--use_gpu", dest="use_gpu", action='store_true')
    parser.add_argument("--use_trt", dest="use_trt", action='store_true')
    parser.add_argument("--use_xla", dest="use_xla", action='store_true')

    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--warmup_times", type=int, default=5, help="warmup")
    parser.add_argument("--repeats", type=int, default=1000, help="repeats")

    return parser.parse_args()

def prepare_config(args):
    config = tf.ConfigProto(log_device_placement=True)
    if args.use_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    return config

def summary_config(args, infer_time : float):
    """
    Args:
        config : paddle inference config
        args : input args
        infer_time : inference time
    """
    logger.info("----------------------- Model info ----------------------")
    logger.info("Model name: {0}, Model type: {1}".format(args.model_name,
                                                          args.model_type))
    logger.info("----------------------- Data info -----------------------")
    logger.info("Batch size: {0}, Num of samples: {1}".format(args.batch_size,
                                                              args.repeats))
    logger.info("----------------------- Conf info -----------------------")
    logger.info("device: {0}".format("gpu" if args.use_gpu else "cpu"))
    if (args.use_gpu):
        logger.info("enable_tensorrt: {0}".format(args.use_trt))
        if (args.use_trt):
            logger.info("trt_precision: {0}".format(args.trt_precision))
    logger.info("enable_xla: {0}".format(args.use_xla))
    logger.info("----------------------- Perf info -----------------------")
    logger.info("Average latency(ms): {0}, QPS: {1}".format(infer_time / args.repeats,
                                    (args.repeats * args.batch_size)/ (infer_time/1000)))

