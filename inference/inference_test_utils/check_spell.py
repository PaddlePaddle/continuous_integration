import re
import os
import argparse


keywords_dict = {
    "op": "OP",
    "cudnn": "cuDNN",
    "tensorrt": "TensorRT",
    "TRT": "TensorRT",
    "cuda": "CUDA",
    "c\\+\\+": "C++",
    "python": "Python",
    "INTEL": "intel",
    "OneDNN": "oneDNN",
}


def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed_files", type=str, default="./changed_files.txt",
                        help="file contains all changed files in current PR")
    return parser.parse_args()


def check_file(file_path: str):
    failed_lines = []
    if file_path.endswith("en.md"):
        pattern_prefix = "\\b"
        pattern_suffix = "\\b"
    else:
        pattern_prefix = pattern_suffix = ""

    with open(file_path, "r") as file:
        data_list = [line.strip() for line in file.readlines()]

    cursor = 0
    while cursor < len(data_list):
        line_num = cursor + 1
        line = data_list[cursor]

        # skip code block
        if line.startswith("```"):
            cursor += 1
            while not data_list[cursor].startswith("```"):
                cursor += 1
            line_num = cursor + 1
            line = data_list[cursor]

        for k, v in keywords_dict.items():
            if re.search(f"{pattern_prefix}{k}{pattern_suffix}", line) is not None:
                failed_lines.append(f"line {line_num}: '{line}' contains non-standard word '{k}' should be '{v}'")

        cursor += 1

    return failed_lines


def main():
    all_failed_lines = []
    with open(args.changed_files, "r") as f:
        file_list = [file.strip() for file in f.readlines()]
    for single_file in file_list:
        if single_file.endswith(".md") or single_file.endswith(".rst"):
            failed_lines = check_file(single_file)
            all_failed_lines.extend(failed_lines)
            print(single_file)
            for line in failed_lines:
                print(line)
            os.system(f"grep -nr '预测' {single_file}")

    if all_failed_lines:
        exit(8)


if __name__ == '__main__':
    args = parse_args()
    main()
