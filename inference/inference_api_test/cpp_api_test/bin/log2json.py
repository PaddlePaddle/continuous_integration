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
import sys
import json
import time
import argparse
import requests


def parse_args():
    """ parse input args """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="log file")
    parser.add_argument("--url", type=str, help="url")
    parser.add_argument("--build_type", type=str, help="build_type")
    parser.add_argument("--repo", type=str, help="repo name")
    parser.add_argument("--inference_path", type=str, help="inference path")
    parser.add_argument("--branch", type=str, help="branch")
    parser.add_argument("--task_type", type=str, help="task type")
    parser.add_argument("--task_name", type=str, help="task name")
    parser.add_argument("--owner", type=str, help="owner")
    parser.add_argument("--build_id", type=str, help="build_id")
    parser.add_argument("--build_number", type=str, help="build_number")

    return parser.parse_args()


def read_log(path):
    key_words = ["FAILED", "finish", "start"]
    key_log = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            for key_word in key_words:
                if "benchmark" in line:
                    continue
                if key_word in line:
                    key_log.append(line.replace('\n', ''))
                else:
                    continue
    return key_log


def josn_file(key_log):
    failed_num = 0
    id = 0
    length = len(key_log)
    josn_list = []
    while id < length:
        line = key_log[id]
        if "start" in line:
            information = line.split()
            name = information[2]
            if "finish" not in key_log[id + 1]:
                status = "failed"
                failed_num = failed_num + 1
                while True:
                    id = id + 1
                    if "finish" not in key_log[id]:
                        continue
                    else:
                        break
            else:
                status = "passed"
            full_name = "cpp_test_" + name
            description = line
            josn_file = {
                "name": name,
                "status": status,
                "fullName": full_name,
                "description": description
            }
            josn_list.append(josn_file)
        id = id + 1
    return josn_list, failed_num


def read_commit_id(inference_path):
    version_path = os.path.join(inference_path, "version.txt")
    with open(version_path) as f:
        first_line = f.readlines()[0]
        f.close()
    return first_line.split()[-1]


def send(args, josn_file, failed_num, commit_id):
    if failed_num > 0:
        status = "Failed"
    else:
        status = "Passed"
    params = {
        "build_type": args.build_type,
        "repo": args.repo,
        "commit_id": commit_id,
        "branch": args.branch,
        "task_type": args.task_type,
        "task_name": args.task_name,
        "owner": args.owner,
        "build_id": args.build_id,
        "build_number": args.build_number,
        "status": status,
        "create_time": time.time(),
        "duration": None,
        "case_detail": json.dumps(josn_file)
    }
    res = requests.post(args.url, data=params)
    print(res.content)


if __name__ == '__main__':
    args = parse_args()
    path = args.input_file
    inference_path = args.inference_path
    commit_id = read_commit_id(inference_path=inference_path)
    key_log = read_log(path=path)
    json_file, failed_num = josn_file(key_log=key_log)
    send(args, json_file, failed_num, commit_id)
