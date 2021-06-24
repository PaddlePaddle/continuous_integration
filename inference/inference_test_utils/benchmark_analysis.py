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
        "--output_name",
        type=str,
        default="benchmark_excel.xlsx",
        help="output excel file name")
    parser.add_argument(
        "--output_html_name",
        type=str,
        default="benchmark_data.html",
        help="output html file name")
    return parser.parse_args()


class BenchmarkLogAnalyzer(object):
    def __init__(self, args):
        """
        """
        self.args = args

        # PaddleInferBenchmark Log Analyzer version, should be same as PaddleInferBenchmark Log Version
        self.analyzer_version = "1.0.3"

        # init dataframe and dict style
        df_columns = [
            "model_name", "paddle_version", "paddle_commit", "paddle_branch",
            "runtime_device", "ir_optim", "enable_memory_optim",
            "enable_tensorrt", "enable_mkldnn", "cpu_math_library_num_threads",
            "precision", "batch_size", "input_shape", "data_num", "cpu_rss(MB)",
            "cpu_vms", "cpu_shared_mb", "cpu_dirty_mb", "cpu_util",
            "gpu_rss(MB)", "gpu_util", "gpu_mem_util", "preprocess_time(ms)",
            "inference_time(ms)", "postprocess_time(ms)"
        ]
        self.benchmark_key = dict.fromkeys(df_columns, None)

        df_columns.remove("paddle_version")
        df_columns.remove("paddle_commit")
        df_columns.remove("paddle_branch")

        df_columns.insert(1, "paddle_info")  # insert to 1 posistion

        self.origin_df = pd.DataFrame(columns=df_columns)

        # merged columns
        self.paddle_info = ["paddle_version", "paddle_commit", "paddle_branch"]

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
        process single inference log
        """
        output_dict = deepcopy(self.benchmark_key)
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
        paddle_info_str = r""
        for k, _ in output_dict.items():
            if not output_dict[k]:
                output_dict[k] = None
                empty_values.append(k)
            if k in self.paddle_info:
                out_str = "".join([str(k), ": ", str(output_dict[k])])
                paddle_info_str = paddle_info_str + out_str + "\n"
        output_dict["paddle_info"] = paddle_info_str
        for paddle_info_key in self.paddle_info:
            output_dict.pop(paddle_info_key)

        if not empty_values:
            logger.info("no empty value found")
        else:
            logger.warning(f"{empty_values} is empty, not found in logs")
        return output_dict

    def __call__(self, log_path, to_database=False):
        """
        """
        # analysis log to dict and dataframe
        for file_name, full_path in self.find_all_logs(log_path):
            dict_log = self.process_log(full_path)
            self.origin_df = self.origin_df.append(dict_log, ignore_index=True)

        raw_df = self.origin_df.sort_values(by='model_name')
        raw_df.sort_values(by=["model_name", "batch_size"], inplace=True)
        raw_df.to_excel(self.args.output_name, index=False)  # render excel
        # convert '\n' to '<br/>' for further usage
        raw_df["paddle_info"] = raw_df["paddle_info"].apply(
            lambda x: x.replace("\n", "<br/>"))
        raw_df.to_html(self.args.output_html_name, index=False)  # render html
        print(raw_df)


def main():
    """
    main
    """
    args = parse_args()
    analyzer = BenchmarkLogAnalyzer(args)
    analyzer(args.log_path, True)


if __name__ == "__main__":
    main()
