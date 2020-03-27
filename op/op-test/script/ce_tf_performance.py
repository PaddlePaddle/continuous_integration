# coding: utf-8

import sys
import os
import json


def run(directory):
    ce_path = "../ce/{}".format(directory)
    file_list = os.listdir(os.path.join(ce_path, "result"))
    for _file in file_list:
        if _file.endswith(".py"):
            os.system("python {} both".format(os.path.join(ce_path, "result", _file)))
            os.system("python {} forward".format(os.path.join(ce_path, "result", _file)))


def generate_result(directory):
    ce_path = "../ce/{}".format(directory)
    file_list = os.listdir(os.path.join(ce_path, "result"))
    with open(os.path.join(ce_path, directory), "w") as fout:
        for _file in file_list:
            if _file.endswith(".result"):
                with open(os.path.join(ce_path, "result", _file), "r") as fin:
                    result = json.load(fin)
                fout.write("{}\t{}\t{}\n".format(_file[:-7], result.get("tf_performance", "-"), result.get("tf_performance_forward", "-")))


if __name__ == "__main__":
    directory = sys.argv[1]
    run(directory)
    generate_result(directory)
