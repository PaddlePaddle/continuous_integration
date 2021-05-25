# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import re
import argparse
import json
import logging
import requests

from copy import deepcopy

import pandas as pd

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    # for local excel analysis
    parser.add_argument(
        "--log_path", type=str, default="./log", help="benchmark log path")

    parser.add_argument(
        "--device_name",
        type=str,
        default=None,
        help="device name, e.g. gpu_t4, gpu_p4")

    # for benchmark platform
    parser.add_argument(
        "--post_url",
        type=str,
        default=None,
        help="post requests url, None will not post to benchmark platform")
    parser.add_argument(
        "--output_json_file",
        type=str,
        default="./post_data.json",
        help="post requests json, cannot be none")
    # basic arguments for framework
    parser.add_argument(
        "--frame_name", type=str, default="paddle", help="framework name")
    parser.add_argument("--api", type=str, default="cpp", help="test api")
    parser.add_argument(
        "--framework_version",
        type=str,
        default="0.0.0",
        help="framework version")
    parser.add_argument(
        "--model_version", type=str, default="0.0.0", help="model version")
    parser.add_argument(
        "--cuda_version", type=str, default="0.0.0", help="cuda version")
    parser.add_argument(
        "--cudnn_version", type=str, default="0.0", help="cudnn version")
    parser.add_argument(
        "--trt_version", type=str, default="0.0.0", help="TensorRT version")
    return parser.parse_args()


class BenchmarkLogBackend(object):
    def __init__(self, args):
        """
        __init__
        """
        self.args = args

        # PaddleInferBenchmark Log Backend version, should be same as PaddleInferBenchmark Log Version
        self.analyzer_version = "1.0.3"

        database_key = [
            'frame_name', 'api', 'model_name', 'model_type', 'batch_size',
            'num_samples', 'framwork_branch', 'model_branch',
            'framework_version', 'model_version', 'cuda_version',
            'cudnn_version', 'trt_version', 'device', 'ir_optim',
            'enable_tensorrt', 'enable_mkldnn', 'trt_precision',
            'cpu_math_library_num_threads', 'average_latency_ms', 'qps',
            'cpu_rss_mb', 'cpu_vms_mb', 'cpu_shared_mb', 'cpu_dirty_mb',
            'cpu_util', 'gpu_used_mb', 'gpu_util', 'gpu_mem_util',
            'preprocess_time_ms', 'postprocess_time_ms', 'inference_time_ms_90',
            'inference_time_ms_99', 'latency_variance'
        ]
        self.database_dict = dict.fromkeys(database_key, None)

        log_key = [
            "paddle_version", "paddle_commit", "paddle_branch",
            "runtime_device", "ir_optim", "enable_memory_optim",
            "enable_tensorrt", "enable_mkldnn", "cpu_math_library_num_threads",
            "model_name", "precision", "batch_size", "input_shape", "data_num",
            "cpu_rss(MB)", "cpu_vms", "cpu_shared_mb", "cpu_dirty_mb",
            "cpu_util", "gpu_rss(MB)", "gpu_util", "gpu_mem_util",
            "preprocess_time(ms)", "inference_time(ms)", "postprocess_time(ms)"
        ]
        self.log_dict = dict.fromkeys(log_key, None)

        if self.args.device_name:
            runtime_device_name = self.args.device_name.split('_')[0]
            if runtime_device_name in ["cpu", "gpu", "xpu"]:
                self.runtime_device_name = self.args.device_name

    def find_all_logs(self, path_walk: str):
        """
        find all .log files from target dir
        """
        for root, ds, files in os.walk(path_walk):
            for file_name in files:
                if re.match(r'.*.log', file_name):
                    full_path = os.path.join(root, file_name)
                    yield file_name, full_path

    def process_log(self, file_name: str) -> dict:
        """
        process log
        """
        output_dict = deepcopy(self.log_dict)
        with open(file_name, 'r') as f:
            for i, data in enumerate(f.readlines()):
                if i == 0:
                    continue
                line_lists = data.split(" ")

                for key_name, _ in output_dict.items():
                    key_name_in_log = "".join([key_name, ":"])
                    if key_name_in_log in line_lists:
                        pos_buf = line_lists.index(key_name_in_log)
                        output_dict[key_name] = line_lists[pos_buf + 1].strip(
                        ).split(',')[0]

        empty_values = []
        for k, _ in output_dict.items():
            if not output_dict[k]:
                output_dict[k] = None
                empty_values.append(k)

        if not empty_values:
            logger.info("no empty value found")
        else:
            logger.warning(f"{empty_values} is empty, not found in logs")
        return output_dict

    def post_key_modify(self, old_log_dict):
        """
        map log key to database key
        """
        new_database_dict = deepcopy(self.database_dict)

        # basic info
        new_database_dict["frame_name"] = self.args.frame_name
        new_database_dict["api"] = self.args.api

        # version info
        new_database_dict["framework_version"] = self.args.framework_version
        new_database_dict["model_version"] = self.args.model_version
        new_database_dict["cuda_version"] = self.args.cuda_version
        new_database_dict["cudnn_version"] = self.args.cudnn_version
        new_database_dict["trt_version"] = self.args.trt_version

        new_database_dict["framwork_branch"] = old_log_dict.get("paddle_branch",
                                                                "")
        new_database_dict["model_branch"] = old_log_dict.get("model_branch", "")

        # model info
        new_database_dict["model_name"] = old_log_dict.get("model_name", "")
        new_database_dict["model_type"] = old_log_dict.get("model_type", "")
        new_database_dict["batch_size"] = old_log_dict.get("batch_size", "")
        new_database_dict["num_samples"] = old_log_dict.get("data_num", "")

        # device
        if old_log_dict['runtime_device'] in self.runtime_device_name:
            new_database_dict["device"] = self.runtime_device_name
        else:
            new_database_dict["device"] = old_log_dict.get("runtime_device", "")

        # config
        new_database_dict["ir_optim"] = old_log_dict.get("ir_optim", "")
        new_database_dict["enable_tensorrt"] = old_log_dict.get(
            "enable_tensorrt", "")
        new_database_dict["enable_mkldnn"] = old_log_dict.get("enable_mkldnn",
                                                              "")
        new_database_dict["trt_precision"] = old_log_dict.get("precision",
                                                              "")
        new_database_dict["cpu_math_library_num_threads"] = old_log_dict.get(
            "cpu_math_library_num_threads", "")

        # performace
        new_database_dict["average_latency_ms"] = old_log_dict.get(
            "inference_time(ms)", "")
        new_database_dict["qps"] = old_log_dict.get("qps", "")

        # memory
        new_database_dict["cpu_rss_mb"] = old_log_dict.get("cpu_rss(MB)", "")
        new_database_dict["cpu_vms_mb"] = old_log_dict.get("cpu_vms_mb", "")
        new_database_dict["cpu_shared_mb"] = old_log_dict.get("cpu_shared_mb",
                                                              "")
        new_database_dict["cpu_dirty_mb"] = old_log_dict.get("cpu_dirty_mb", "")
        new_database_dict["cpu_util"] = old_log_dict.get("cpu_util", "")
        new_database_dict["gpu_used_mb"] = old_log_dict.get("gpu_rss(MB)", "")
        new_database_dict["gpu_util"] = old_log_dict.get("gpu_util", "")
        new_database_dict["gpu_mem_util"] = old_log_dict.get("gpu_mem_util", "")

        # perfomance
        new_database_dict["preprocess_time_ms"] = old_log_dict.get(
            "preprocess_time_ms", "")
        new_database_dict["postprocess_time_ms"] = old_log_dict.get(
            "postprocess_time_ms", "")
        new_database_dict["inference_time_ms_90"] = old_log_dict.get(
            "inference_time_ms_90", "")
        new_database_dict["inference_time_ms_99"] = old_log_dict.get(
            "inference_time_ms_99", "")
        new_database_dict["latency_variance"] = old_log_dict.get(
            "latency_variance", "")

        return new_database_dict

    def __call__(self, log_path):
        """
        __call__
        """
        # analysis log to dict and dataframe
        json_list = []
        for file_name, full_path in self.find_all_logs(log_path):
            dict_log = self.process_log(full_path)

            new_dict_log = self.post_key_modify(dict_log)
            json_list.append(new_dict_log)
            with open(self.args.output_json_file, 'w') as f:
                json.dump(json_list, f)

        if not os.path.exists(self.args.output_json_file):
            raise ValueError("{} has not been created".format(
                self.args.output_json_file))

        # post request
        files = {'file': open(self.args.output_json_file, 'rb')}
        response = requests.post(self.args.post_url, files=files)
        logger.info(response)
        assert response.status_code == 200, f"send post request failed, please check input data structure and post urls, \njson is {json_list}"
        logger.info(
            "==== post request succeed, response status_code is {} ====".format(
                response.status_code))


def main():
    """
    main
    """
    args = parse_args()
    analyzer = BenchmarkLogBackend(args)
    analyzer(args.log_path)


if __name__ == "__main__":
    main()
