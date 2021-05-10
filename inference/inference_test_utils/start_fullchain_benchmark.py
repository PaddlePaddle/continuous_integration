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
import argparse
import hashlib

import wget
from git import Repo

def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def prepare_repo():
    """
    download specific repo codes
    but if code exists, don't download
    """
    if os.path.exists('./PaddleOCR'):
        print("Paddle OCR repo has been downloaded")
    else:
        repo = Repo.clone_from(
            'https://github.com/PaddlePaddle/PaddleOCR.git',
            './PaddleOCR',
            branch='release/2.1')
        cmd1 = "git fetch origin pull/2308/head:pr_test_branch"
        os.system(cmd1)
        cmd2 = "git checkout pr_test_branch"
        os.system(cmd2)

def md5_check(file_name: str, original_md5: str) -> bool:
    """
    check file md5
    file_name = './PaddleOCR/tools/infer/utility.py'
    original_md5 = 'a22a438726009aabb94382d248313e7b'
    """
    # Open,close, read file and calculate MD5 on its contents
    md5_hash = hashlib.md5()
    with open(file_name) as file_to_check:
        # read contents of the file and encode it with utf-8
        content = file_to_check.read().encode('utf-8')
        # pipe contents of the file through
        md5_hash.update(content)
        md5_returned = md5_hash.hexdigest()

    # Finally compare original MD5 with freshly calculated
    if original_md5 == md5_returned:
        print("MD5 verified.")
        return True
    else:
        print("MD5 verification failed!.")
        return False

def prepare_test_codes():
    """
    prepare test codes
    if test file md5 correct, don't download or replace
    """
    logger_file = "fullchain_log.py"
    logger_file_md5 = "9d1af35fe0acb2ad7dc7f58e90f968d0"

    if os.path.exists(logger_file) and md5_check(logger_file, logger_file_md5):
        print("fullchain_log.py file exist, and md5sum correct")
    else:
        logger_file_url = "https://raw.githubusercontent.com/OliverLPH/continuous_integration/fullchain_log/inference/inference_test_utils/fullchain_log.py"
        wget.download(logger_file_url, out="./")

def paddle_train_model():
    """
    train paddle model for inference
    """
    pass

def paddle_infer_benchmark():
    """
    start run paddle inference benchmark
    """
    cmd = "bash ./PaddleOCR/benchmark.sh"
    os.system(cmd)

def paddle_serving_benchmark():
    """
    start runn paddle serving benchmark
    """
    pass


def paddle_lite_benchmark():
    """
    start run paddle lite benchmark
    """
    pass


def summary_benchmark_data():
    """
    batch summary benchmark data
    """
    cmd = "python py_fullchain_analysis.py --log_path=./log_path \
                                           --output_name=test_benhmark_excel.xlsx"


def main():
    """
    main
    """
    os.environ["http_proxy"] = "http://172.19.57.45:3128"
    os.environ["https_proxy"] = "http://172.19.57.45:3128"
    prepare_repo()
    prepare_test_codes()
    # paddle_infer_benchmark()
    # summary_benchmark_data()


if __name__ == "__main__":
    main()
