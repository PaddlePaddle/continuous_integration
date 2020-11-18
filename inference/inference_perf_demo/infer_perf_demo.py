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
import numpy as np
import paddle.fluid.inference as paddle_infer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--model_type", type=str, help="model generate type")
    parser.add_argument("--model_file", type=str, help="model filename")
    parser.add_argument("--params_file", type=str, help="parameter filename")

    parser.add_argument("--use_gpu", dest="use_gpu", action='store_true')
    parser.add_argument("--use_trt", dest="use_trt", action='store_true')
    parser.add_argument("--use_mkldnn", dest="use_mkldnn", action='store_true')

    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--warmup_times", type=int, default=0, help="warmup")
    parser.add_argument("--repeats", type=int, default=1, help="repeats")
    parser.add_argument(
        "--cpu_math_library_num_threads",
        type=int,
        default=1,
        help="math_thread_num")

    return parser.parse_args()


def prepare_config(args):
    config = paddle_infer.Config(args.model_file, args.params_file)
    if (args.use_gpu):
        config.enable_use_gpu(100, 0)
        if (args.use_trt):
            config.enable_tensorrt_engine()
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(
            args.cpu_math_library_num_threads)
        if (args.use_mkldnn):
            config.enable_mkldnn()
    return config


def Inference(args, predictor):
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    fake_input = np.ones((args.batch_size, 3, 224, 224)).astype("float32")
    input_handle.reshape([args.batch_size, 3, 224, 224])
    input_handle.copy_from_cpu(fake_input)

    for i in range(args.warmup_times):
        predictor.run()

    time1 = time.time()
    for i in range(args.repeats):
        predictor.run()
        output_names = predictor.get_output_names()
        output_hanlde = predictor.get_output_handle(output_names[0])
    time2 = time.time()
    output_data = output_hanlde.copy_to_cpu()

    total_inference_cost = (time2 - time1) * 1000  # total latency, ms

    return total_inference_cost


def run_demo():
    args = parse_args()
    config = prepare_config(args)
    predictor_pool = paddle_infer.PredictorPool(config, 1)
    predictor = predictor_pool.retrive(0)
    total_time = Inference(args, predictor)

    print("----------------------- Model info ----------------------")
    print("Model name: {}, Model type: {}".format(args.model_name,
                                                  args.model_type))
    print("----------------------- Data info -----------------------")
    print("Batch size: {}, Num of samples: {}".format(args.batch_size,
                                                      args.repeats))
    print("----------------------- Conf info -----------------------")
    print("device: {}, ir_optim: {}".format("gpu" if config.use_gpu() else
                                            "cpu", config.ir_optim()))
    if (args.use_gpu):
        if (args.use_trt):
            print("enable_tensorrt: {}".format(config.tensorrt_engine_enabled(
            )))
    else:
        print("cpu_math_library_num_threads: {}".format(
            config.cpu_math_library_num_threads()))
        if (args.use_mkldnn):
            print("enable_mkldnn: {}".format(config.mkldnn_enabled()))
    print("----------------------- Perf info -----------------------")
    print("Average latency(ms): {}, QPS: {}".format(total_time / args.repeats, (
        args.repeats * args.batch_size) / (total_time / 1000)))


if __name__ == "__main__":
    run_demo()
