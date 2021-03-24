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
import os
import re
import json
import logging

import requests

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    # log arguments
    parser.add_argument(
        "--log_path",
        type=str,
        default="./log",
        help="benchmark log path, should be directory")
    parser.add_argument(
        "--post_url", type=str, help="post requests url, cannot be none")
    parser.add_argument(
        "--output_json_file",
        type=str,
        default="./post_data.json",
        help="post requests url, cannot be none")

    # framework basic arguments
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


def find_all_logs(path_walk: str):
    """
    find all .log files from target dir
    """
    for root, ds, files in os.walk(path_walk):
        for file_name in files:
            if re.match(r'.*.log', file_name):
                full_path = os.path.join(root, file_name)
                yield file_name, full_path


def process_log(file_name: str) -> dict:
    """
    process log to dict
    """
    output_dict = {}
    with open(file_name, 'r') as f:
        for i, data in enumerate(f.readlines()):
            if i == 0:
                continue
            line_lists = data.split(" ")
            if "name:" in line_lists and "type:" in line_lists:
                pos_buf = line_lists.index("name:")
                output_dict["model_name"] = line_lists[pos_buf + 1].split(',')[
                    0]
                output_dict["model_type"] = line_lists[-1].strip()
            if "Num" in line_lists and "size:" in line_lists:
                pos_buf = line_lists.index("size:")
                output_dict["batch_size"] = line_lists[pos_buf + 1].split(',')[
                    0]
                output_dict["num_samples"] = line_lists[-1].strip()
            if "ir_optim:" in line_lists and "device:" in line_lists:
                pos_buf = line_lists.index("ir_optim:")
                output_dict["device"] = line_lists[pos_buf - 1].split(',')[0]
                output_dict["ir_optim"] = line_lists[-1].strip()
            if "enable_tensorrt:" in line_lists:
                output_dict["enable_tensorrt"] = line_lists[-1].strip()
            if "QPS:" in line_lists and "latency(ms):" in line_lists:
                pos_buf = line_lists.index("QPS:")
                output_dict["average_latency"] = line_lists[pos_buf - 1].split(
                    ',')[0]
                output_dict["qps"] = line_lists[-1].strip()
            if "enable_mkldnn:" in line_lists:
                output_dict["enable_mkldnn"] = line_lists[-1].strip()
            if "cpu_math_library_num_threads:" in line_lists:
                output_dict["cpu_math_library_num_threads"] = line_lists[
                    -1].strip()
            if "trt_precision:" in line_lists:
                output_dict["trt_precision"] = line_lists[-1].strip()
            if "rss(MB):" in line_lists and "cpu_usage(%):" in line_lists:
                pos_buf = line_lists.index("vms(MB):")
                output_dict["cpu_rss"] = line_lists[pos_buf - 1].split(',')[0]
                output_dict["cpu_vms"] = line_lists[pos_buf + 1].split(',')[0]
                output_dict["cpu_shared"] = line_lists[pos_buf + 3].split(',')[
                    0]
                output_dict["cpu_dirty"] = line_lists[pos_buf + 5].split(',')[0]
                output_dict["cpu_usage"] = line_lists[pos_buf + 7].split(',')[0]
            if "total(MB):" in line_lists and "free(MB):" in line_lists:
                pos_buf = line_lists.index("gpu_utilization_rate(%):")
                output_dict["gpu_used"] = line_lists[pos_buf - 1].split(',')[0]
                output_dict["gpu_utilization_rate"] = line_lists[
                    pos_buf + 1].split(',')[0]
                output_dict["gpu_mem_utilization_rate"] = line_lists[
                    pos_buf + 3].split(',')[0]
    return output_dict


def main():
    """
    main, process structured log and post
    """
    args = parse_args()
    if not args.post_url:
        raise ValueError(
            "post_url has not been defined, please pass post_url into argument")

    if not os.path.exists(args.log_path):
        raise ValueError("{} does not exists, no log will be processed".format(
            args.log_path))

    json_list = []
    for file_name, full_path in find_all_logs(args.log_path):
        # logger.info(file_name, full_path)
        dict_log = process_log(full_path)
        # basic info
        dict_log["frame_name"] = args.frame_name
        dict_log["api"] = args.api

        # version info
        dict_log["framework_version"] = args.framework_version
        dict_log["model_version"] = args.model_version
        dict_log["cuda_version"] = args.cuda_version
        dict_log["cudnn_version"] = args.cudnn_version
        dict_log["trt_version"] = args.trt_version

        # append dict log
        json_list.append(dict_log)

    with open(args.output_json_file, 'w') as f:
        json.dump(json_list, f)

    if not os.path.exists(args.output_json_file):
        raise ValueError("{} has not been created".format(
            args.output_json_file))

    # post request
    files = {'file': open(args.output_json_file, 'rb')}
    response = requests.post(args.post_url, files=files)
    logger.info(response)
    assert response.status_code == 200, "send post request failed, please check input data structure and post urls"
    logger.info("==== post request succeed, response status_code is {} ====".
                format(response.status_code))


if __name__ == "__main__":
    main()
