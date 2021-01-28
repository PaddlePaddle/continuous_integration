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
                output_dict["gpu_used(MB):"] = line_lists[pos_buf - 1].split(',')[0]
                output_dict["gpu_utilization_rate(%)"] = line_lists[pos_buf + 1].split(',')[0]
                output_dict["gpu_mem_utilization_rate(%)"] = line_lists[pos_buf + 3].split(',')[0]
    return output_dict

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
                                      "cpu_usage(%)", "gpu_used(MB):",
                                      "gpu_utilization_rate(%)",
                                      "gpu_mem_utilization_rate(%)"])

    for file_name, full_path in find_all_logs(args.log_path):
        # print(file_name, full_path)
        dict_log = process_log(full_path)
        origin_df = origin_df.append(dict_log, ignore_index=True)

    saved_df = origin_df.sort_values(by='model_name')
    saved_df.to_excel(args.output_name)
    print(saved_df)

if __name__ == "__main__":
    main()

