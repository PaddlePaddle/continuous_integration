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
        "--analysis_trt", dest="analysis_trt", action='store_true')
    parser.add_argument(
        "--analysis_mkl", dest="analysis_mkl", action='store_true')

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

            # conf info
            if "runtime_device:" in line_lists:
                pos_buf = line_lists.index("runtime_device:")
                output_dict["runtime_device"] = line_lists[pos_buf + 1].strip()
            if "ir_optim:" in line_lists:
                pos_buf = line_lists.index("ir_optim:")
                output_dict["ir_optim"] = line_lists[pos_buf + 1].strip()
            if "enable_memory_optim:" in line_lists:
                pos_buf = line_lists.index("enable_memory_optim:")
                output_dict["enable_memory_optim"] = line_lists[pos_buf +
                                                                1].strip()
            if "enable_tensorrt:" in line_lists:
                pos_buf = line_lists.index("enable_tensorrt:")
                output_dict["enable_tensorrt"] = line_lists[pos_buf + 1].strip()
            if "precision:" in line_lists:
                pos_buf = line_lists.index("precision:")
                output_dict["precision"] = line_lists[pos_buf + 1].strip()
            if "enable_mkldnn:" in line_lists:
                pos_buf = line_lists.index("enable_mkldnn:")
                output_dict["enable_mkldnn"] = line_lists[pos_buf + 1].strip()
            if "cpu_math_library_num_threads:" in line_lists:
                pos_buf = line_lists.index("cpu_math_library_num_threads:")
                output_dict["cpu_math_library_num_threads"] = line_lists[
                    pos_buf + 1].strip()

            # model info
            if "model_name:" in line_lists:
                pos_buf = line_lists.index("model_name:")
                output_dict["model_name"] = list(
                    filter(None, line_lists[pos_buf + 1].strip().split('/')))[
                        -1]

            # data info
            if "batch_size:" in line_lists:
                pos_buf = line_lists.index("batch_size:")
                output_dict["batch_size"] = line_lists[pos_buf + 1].strip()
            if "input_shape:" in line_lists:
                pos_buf = line_lists.index("input_shape:")
                output_dict["input_shape"] = line_lists[pos_buf + 1].strip()
            if "data_num:" in line_lists:
                pos_buf = line_lists.index("data_num:")
                output_dict["data_num"] = line_lists[pos_buf + 1].strip()

            # perf info
            if "cpu_rss(MB):" in line_lists:
                pos_buf = line_lists.index("cpu_rss(MB):")
                output_dict["cpu_rss(MB)"] = line_lists[pos_buf + 1].strip(
                ).split(',')[0]
            if "gpu_rss(MB):" in line_lists:
                pos_buf = line_lists.index("gpu_rss(MB):")
                output_dict["gpu_rss(MB)"] = line_lists[pos_buf + 1].strip(
                ).split(',')[0]
            if "gpu_util:" in line_lists:
                pos_buf = line_lists.index("gpu_util:")
                output_dict["gpu_util"] = line_lists[pos_buf + 1].strip().split(
                    ',')[0]
            if "preprocess_time(ms):" in line_lists:
                pos_buf = line_lists.index("preprocess_time(ms):")
                output_dict["preprocess_time(ms)"] = line_lists[
                    pos_buf + 1].strip().split(',')[0]
            if "inference_time(ms):" in line_lists:
                pos_buf = line_lists.index("inference_time(ms):")
                output_dict["inference_time(ms)"] = line_lists[
                    pos_buf + 1].strip().split(',')[0]
            if "postprocess_time(ms):" in line_lists:
                pos_buf = line_lists.index("postprocess_time(ms):")
                output_dict["postprocess_time(ms)"] = line_lists[
                    pos_buf + 1].strip().split(',')[0]
    return output_dict


def filter_df_merge(cpu_df, filter_column=None):
    """
    process cpu data frame, merge by 'model_name', 'batch_size'
    Args:
        cpu_df ([type]): [description]
    """
    if not filter_column:
        raise Exception(
            "please assign filter_column for filter_df_merge function")

    df_lists = []
    filter_column_lists = []
    output_df_has_nan = False
    for k, v in cpu_df.groupby(filter_column, dropna=True):
        # True, False, Nan
        # print(k, v) # not use list generator here
        filter_column_lists.append(k)
        df_lists.append(v)
    final_output_df = df_lists[-1]

    # merge same model
    for i in range(len(df_lists) - 1):
        left_suffix = cpu_df[filter_column].unique()[0]
        right_suffix = df_lists[i][filter_column].unique()[0]
        # enable_mkldnn = [True, False]
        # cpu_math_library_num_threads = [6, 1]
        print(left_suffix, right_suffix)
        print('========================')
        if not pd.isnull(right_suffix):
            final_output_df = pd.merge(
                final_output_df,
                df_lists[i],
                how='left',
                left_on=['model_name', 'batch_size'],
                right_on=['model_name', 'batch_size'],
                suffixes=('', '_{0}_{1}'.format(filter_column, right_suffix)))
        else:
            pass

    # rename default df columns
    origin_column_names = list(cpu_df.columns.values)
    origin_column_names.remove(filter_column)
    suffix = final_output_df[filter_column].unique()[0]
    for name in origin_column_names:
        final_output_df.rename(
            columns={name: "{0}_{1}_{2}".format(name, filter_column, suffix)},
            inplace=True)
    final_output_df.rename(
        columns={
            filter_column: "{0}_{1}_{2}".format(filter_column, filter_column,
                                                suffix)
        },
        inplace=True)

    final_output_df.sort_values(
        by=[
            "model_name_{0}_{1}".format(filter_column, suffix),
            "batch_size_{0}_{1}".format(filter_column, suffix)
        ],
        inplace=True)
    return final_output_df


def trt_perf_analysis(raw_df):
    """
    sperate raw dataframe to a list of dataframe
    compare tensorrt percision performance
    """
    # filter df by gpu, compare tensorrt and gpu
    # define default dataframe for gpu performance analysis
    gpu_df = raw_df.loc[raw_df['runtime_device'] == 'gpu']
    new_df = filter_df_merge(gpu_df, "precision")

    # calculate qps diff percentail
    new_df["fp32_fp16_diff"] = new_df[[
        "inference_time(ms)_{}_fp32".format("precision"),
        "inference_time(ms)_{}_fp16".format("precision")
    ]].apply(
        lambda x: (float(x["inference_time(ms)_{}_fp16".format("precision")]) - float(x["inference_time(ms)_{}_fp32".format("precision")])) / float(x["inference_time(ms)_{}_fp32".format("precision")]),
        axis=1)
    new_df["fp32_gpu_diff"] = new_df[[
        "inference_time(ms)", "inference_time(ms)_{}_fp32".format("precision")
    ]].apply(
        lambda x: (float(x["inference_time(ms)_{}_fp32".format("precision")]) - float(x["inference_time(ms)_{}_fp32".format("precision")])) / float(x["inference_time(ms)".format("precision")]),
        axis=1)
    new_df["fp16_int8_diff"] = new_df[[
        "inference_time(ms)_{}_fp16".format("precision"),
        "inference_time(ms)_{}_int8".format("precision")
    ]].apply(
        lambda x: (float(x["inference_time(ms)_{}_int8".format("precision")]) - float(x["inference_time(ms)_{}_fp16"])) / float(x["inference_time(ms)_{}_fp16".format("precision")]),
        axis=1)

    return new_df


def mkl_perf_analysis(raw_df):
    """
    sperate raw dataframe to a list of dataframe
    compare mkldnn performance with not enable mkldnn
    """
    # filter df by cpu, compare mkl and cpu
    # define default dataframe for cpu mkldnn analysis
    cpu_df = raw_df.loc[raw_df['runtime_device'] == 'cpu']
    mkl_compare_df = cpu_df.loc[cpu_df['cpu_math_library_num_threads'] == '1']
    thread_compare_df = cpu_df.loc[cpu_df['enable_mkldnn'] == 'True']

    # define dataframe need to be analyzed
    output_mkl_df = filter_df_merge(mkl_compare_df, 'enable_mkldnn')
    output_thread_df = filter_df_merge(thread_compare_df,
                                       'cpu_math_library_num_threads')

    # calculate performance diff percentail
    # compare mkl performance with cpu
    output_mkl_df["mkl_infer_diff"] = output_mkl_df[[
        "inference_time(ms)_{}_True".format("enable_mkldnn"),
        "inference_time(ms)_{}_False".format("enable_mkldnn")
    ]].apply(
        lambda x: (float(x["inference_time(ms)_{}_True".format("enable_mkldnn")]) - float(x["inference_time(ms)_{}_False".format("enable_mkldnn")])) / float(x["inference_time(ms)_{}_False".format("enable_mkldnn")]),
        axis=1)
    output_mkl_df["mkl_cpu_rss_diff"] = output_mkl_df[[
        "cpu_rss(MB)_{}_True".format("enable_mkldnn"),
        "cpu_rss(MB)_{}_False".format("enable_mkldnn")
    ]].apply(
        lambda x: (float(x["cpu_rss(MB)_{}_True".format("enable_mkldnn")]) - float(x["cpu_rss(MB)_{}_False".format("enable_mkldnn")])) / float(x["cpu_rss(MB)_{}_False".format("enable_mkldnn")]),
        axis=1)

    # compare cpu_multi_thread performance with cpu
    output_thread_df["mkl_infer_diff"] = output_thread_df[[
        "inference_time(ms)_{}_6".format("cpu_math_library_num_threads"),
        "inference_time(ms)_{}_1".format("cpu_math_library_num_threads")
    ]].apply(
        lambda x: (float(x["inference_time(ms)_{}_6".format("cpu_math_library_num_threads")]) - float(x["inference_time(ms)_{}_1".format("cpu_math_library_num_threads")])) / float(x["inference_time(ms)_{}_1".format("cpu_math_library_num_threads")]),
        axis=1)
    output_thread_df["mkl_cpu_rss_diff"] = output_thread_df[[
        "cpu_rss(MB)_{}_6".format("cpu_math_library_num_threads"),
        "cpu_rss(MB)_{}_1".format("cpu_math_library_num_threads")
    ]].apply(
        lambda x: (float(x["cpu_rss(MB)_{}_6".format("cpu_math_library_num_threads")]) - float(x["cpu_rss(MB)_{}_1".format("cpu_math_library_num_threads")])) / float(x["cpu_rss(MB)_{}_1".format("cpu_math_library_num_threads")]),
        axis=1)

    return output_mkl_df, output_thread_df


def post_benchmark(args):
    """
    post json data to benchmark backend
    """
    if not os.path.exists(args.log_path):
        raise ValueError("{} does not exists, no log will be processed".format(
            args.log_path))

    json_list = []
    for file_name, full_path in find_all_logs(args.log_path):
        print(file_name)
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

        # modify key for upload to benchmark backend
        dict_log["device"] = dict_log.pop("runtime_device")
        dict_log["ir_optim"] = dict_log["ir_optim"].lower()
        dict_log.pop("enable_memory_optim")
        dict_log["enable_tensorrt"] = dict_log["enable_tensorrt"].lower()
        dict_log["enable_mkldnn"] = dict_log["enable_mkldnn"].lower()

        dict_log["cpu_rss"] = dict_log.pop("cpu_rss(MB)")
        dict_log["gpu_used"] = dict_log.pop("gpu_rss(MB)")
        dict_log["gpu_utilization_rate"] = dict_log.pop("gpu_util")
        dict_log.pop("preprocess_time(ms)")
        dict_log["average_latency"] = dict_log.pop("inference_time(ms)")
        dict_log.pop("postprocess_time(ms)")
        dict_log["num_samples"] = dict_log.pop("data_num")
        dict_log["qps"] = str((int(dict_log["num_samples"]) * int(dict_log['batch_size'])) / (float(dict_log["average_latency"]) * float(dict_log["num_samples"]) /1000))
        dict_log["trt_precision"] = dict_log.pop("precision")

        dict_log["cpu_vms"] = '0'
        dict_log["cpu_shared"] = '0'
        dict_log["cpu_usage"] = '0'
        dict_log["gpu_mem_utilization_rate"] = '0'
        dict_log.pop("input_shape")


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


def save_excel(args):
    """
    save benchmark data to excel for analysis
    """
    # create empty DataFrame
    origin_df = pd.DataFrame(columns=[
        "model_name", "batch_size", "input_shape", "runtime_device", "ir_optim",
        "enable_memory_optim", "enable_tensorrt", "precision", "enable_mkldnn",
        "cpu_math_library_num_threads", "preproce_time(ms)",
        "inference_time(ms)", "postprocess_time(ms)", "cpu_rss(MB)",
        "gpu_rss(MB)", "gpu_util"
    ])

    for file_name, full_path in find_all_logs(args.log_path):
        # print(file_name, full_path)
        dict_log = process_log(full_path)
        origin_df = origin_df.append(dict_log, ignore_index=True)

    raw_df = origin_df.sort_values(by='model_name')
    raw_df.sort_values(by=["model_name", "batch_size"], inplace=True)
    raw_df.to_excel(args.output_name)
    print(raw_df)

    if args.analysis_trt:
        trt_df = trt_perf_analysis(raw_df)
        trt_df.to_excel("trt_analysis_{}".format(args.output_name))

    if args.analysis_mkl:
        mkl_df, thread_df = mkl_perf_analysis(raw_df)
        mkl_df.to_excel("mkl_enable_analysis_{}".format(args.output_name))
        thread_df.to_excel("mkl_threads_analysis_{}".format(args.output_name))


def main():
    """
    main
    """
    args = parse_args()
    if not args.post_url:
        logger.info(
            "post_url has not been defined, please pass post_url into argument")
    else:
        post_benchmark(args)

    save_excel(args)


if __name__ == "__main__":
    main()
