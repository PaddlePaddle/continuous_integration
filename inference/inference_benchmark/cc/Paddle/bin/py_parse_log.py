# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import os
import re
import argparse

import pandas as pd

def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="./log",
                        help="benchmark log path")
    parser.add_argument("--output_name", type=str, default="benchmark_excel.xlsx",
                        help="output excel file name")
    parser.add_argument("--process_trt", dest="process_trt", action='store_true')
    return parser.parse_args()


def find_all_logs(path_walk : str):
    """
    find all .log files from target dir
    """
    for root, ds, files in os.walk(path_walk):
        for file_name in files:
            if re.match(r'.*.log', file_name):
                full_path = os.path.join(root, file_name)
                yield file_name, full_path

def process_log(file_name : str) -> dict:
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
                output_dict["model_name"] = line_lists[pos_buf + 1].split(',')[0]
                output_dict["model_type"] = line_lists[-1].strip()
            if "Num" in line_lists and "size:" in line_lists:
                pos_buf = line_lists.index("size:")
                output_dict["batch_size"] = line_lists[pos_buf + 1].split(',')[0]
                output_dict["num_samples"] = line_lists[-1].strip()
            if "ir_optim:" in line_lists and "device:" in line_lists:
                pos_buf = line_lists.index("ir_optim:")
                output_dict["device"] = line_lists[pos_buf - 1].split(',')[0]
                output_dict["ir_optim"] = line_lists[-1].strip()
            if "enable_tensorrt:" in line_lists:
                output_dict["enable_tensorrt"] = line_lists[-1].strip()
            if "QPS:" in line_lists and "latency(ms):" in line_lists:
                pos_buf = line_lists.index("QPS:")
                output_dict["Average_latency"] = line_lists[pos_buf - 1].split(',')[0]
                output_dict["QPS"] = line_lists[-1].strip()
            if "enable_mkldnn:" in line_lists:
                output_dict["enable_mkldnn"] = line_lists[-1].strip()
            if "cpu_math_library_num_threads:" in line_lists:
                output_dict["cpu_math_library_num_threads"] = line_lists[-1].strip()
            if "trt_precision:" in line_lists:
                output_dict["trt_precision"] = line_lists[-1].strip()
            if "rss(MB):" in line_lists and "cpu_usage(%):" in line_lists:
                pos_buf = line_lists.index("vms(MB):")
                output_dict["cpu_rss(MB)"] = line_lists[pos_buf - 1].split(',')[0]
                output_dict["cpu_vms(MB)"] = line_lists[pos_buf + 1].split(',')[0]
                output_dict["cpu_shared(MB)"] = line_lists[pos_buf + 3].split(',')[0]
                output_dict["cpu_dirty(MB)"] = line_lists[pos_buf + 5].split(',')[0]
                output_dict["cpu_usage(%)"] = line_lists[pos_buf + 7].split(',')[0]
            if "total(MB):" in line_lists and "free(MB):" in line_lists:
                pos_buf = line_lists.index("gpu_utilization_rate(%):")
                output_dict["gpu_used(MB)"] = line_lists[pos_buf - 1].split(',')[0]
                output_dict["gpu_utilization_rate(%)"] = line_lists[pos_buf + 1].split(',')[0]
                output_dict["gpu_mem_utilization_rate(%)"] = line_lists[pos_buf + 3].split(',')[0]
    return output_dict

def compare_trt_perf(raw_df):
    """
    sperate raw dataframe to a list of dataframe
    compare tensorrt percision performance
    """
    df_lists = []
    precisions = []
    precision_has_nan = False
    for k,v in raw_df.groupby('trt_precision', dropna=False):
        # fp16, fp32, int8, Nan
        print(k)  # not use list generator here
        precisions.append(k)
        df_lists.append(v)
        if pd.isnull(k):
            precision_has_nan = True
            new_df = v

    # merge fp16, fp32, int8, Nan "trt_precision"
    # new_df = df_lists[-1]
    for i in range(len(df_lists)):
        left_suffix = new_df["trt_precision"].unique()[0]
        right_suffix = df_lists[i]["trt_precision"].unique()[0]
        if not pd.isnull(right_suffix):
            new_df = pd.merge(new_df, df_lists[i],  how='left',
                              left_on=['model_name','batch_size'],
                              right_on = ['model_name','batch_size'],
                              suffixes=('', '_trt{}'.format(right_suffix)))
        else:
            pass

    # calculate qps diff percentail
    if "fp16" in precisions and "fp32" in precisions:
        new_df["fp32_fp16_diff"] = new_df[["QPS_trtfp32", "QPS_trtfp16"]].apply(
            lambda x:(float(x["QPS_trtfp16"]) - float(x["QPS_trtfp32"]))/float(x["QPS_trtfp32"]), axis=1)
    if "fp32" in precisions and precision_has_nan:
        new_df["fp32_gpu_diff"] = new_df[["QPS", "QPS_trtfp32"]].apply(
            lambda x:(float(x["QPS_trtfp32"]) - float(x["QPS"]))/float(x["QPS"]), axis=1)
    if "fp16" in precisions and "int8" in precisions:
        new_df["fp16_int8_diff"] = new_df[["QPS_trtfp16", "QPS_trtint8"]].apply(
            lambda x:(float(x["QPS_trtint8"]) - float(x["QPS_trtfp16"]))/float(x["QPS_trtfp16"]), axis=1)

    return new_df


def main():
    """
    main
    """
    args = parse_args()
    # create empty DataFrame
    origin_df = pd.DataFrame(columns=["model_name", "model_type",
                                      "batch_size", "num_samples",
                                      "device", "ir_optim",
                                      "enable_tensorrt", "enable_mkldnn",
                                      "trt_precision",
                                      "cpu_math_library_num_threads",
                                      "Average_latency", "QPS",
                                      "cpu_rss(MB)", "cpu_vms(MB)",
                                      "cpu_shared(MB)", "cpu_dirty(MB)",
                                      "cpu_usage(%)", "gpu_used(MB)",
                                      "gpu_utilization_rate(%)",
                                      "gpu_mem_utilization_rate(%)"])

    for file_name, full_path in find_all_logs(args.log_path):
        # print(file_name, full_path)
        dict_log = process_log(full_path)
        origin_df = origin_df.append(dict_log, ignore_index=True)

    raw_df = origin_df.sort_values(by='model_name')
    raw_df.to_excel(args.output_name)
    print(raw_df)
    
    if args.process_trt:
        trt_df = compare_trt_perf(raw_df)
        trt_df.to_excel("trt_res_{}".format(args.output_name))
        print(trt_df)


if __name__ == "__main__":
    main()
