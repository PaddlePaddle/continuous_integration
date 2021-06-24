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
        "--log_file", type=str, default="result.log", help="ci result log path")
    return parser.parse_args()


def _find_char(input_char):
    """
    find english char in input string
    """
    result = re.findall(r'[a-zA-Z=_/0-9.]+', str(input_char))
    return result


def process_log(file_name: str):
    """
    process log
    """
    train_list_dict = []
    export_list_dict = []
    predict_det_list_dict = []
    with open(file_name, 'r') as f:
        for i, data in enumerate(f.readlines()):
            # print(i, data)
            train_dict = {}
            if "train.py" in data:
                split_data = data.split(' ')
                for line_value in split_data:
                    if "=" in line_value:
                        key = _find_char(line_value.split('=')[0])
                        value = _find_char(line_value.split('=')[-1])
                        # print(key, value)
                        train_dict[key[0]] = ''.join(value)
                if "successfully" in split_data:
                    train_dict["status"] = "passed"
                else:
                    train_dict["status"] = "failed"
                # print(train_dict)
                train_list_dict.append(train_dict)

            export_dict = {}
            if "export_model.py" in data:
                split_data = data.split(' ')
                for line_value in split_data:
                    if "=" in line_value:
                        key = _find_char(line_value.split('=')[0])
                        value = _find_char(line_value.split('=')[-1])
                        # print(key, value)
                        export_dict[key[0]] = ''.join(value)
                if "successfully" in split_data:
                    export_dict["status"] = "passed"
                else:
                    export_dict["status"] = "failed"
                # print(export_dict)
                export_list_dict.append(export_dict)

            predict_det_dict = {}
            if "predict_det.py" in data:
                split_data = data.split(' ')
                for line_value in split_data:
                    if "=" in line_value:
                        key = _find_char(line_value.split('=')[0])
                        value = _find_char(line_value.split('=')[-1])
                        # print(key, value)
                        predict_det_dict[key[0]] = ''.join(value)
                if "successfully" in split_data:
                    predict_det_dict["status"] = "passed"
                else:
                    predict_det_dict["status"] = "failed"
                # print(predict_det_dict)
                predict_det_list_dict.append(predict_det_dict)
    return train_list_dict, export_list_dict, predict_det_list_dict


def main():
    """
    main
    """
    args = parse_args()
    a, b, c = process_log(args.log_file)
    a_1 = pd.DataFrame(a)
    b_1 = pd.DataFrame(b)
    c_1 = pd.DataFrame(c)
    print(a_1)
    print(b_1)
    print(c_1)

    a_1.to_html("train_ci_log.html", index=False)  # render html
    b_1.to_html("export_ci_log.html", index=False)  # render html
    c_1.to_html("predict_det_ci_log.html", index=False)  # render html


if __name__ == "__main__":
    main()
