#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
auto generate cases
Authors: xuguangyao01(xuguangyao01@baidu.com)
Date:    2019/07/22 14:01:11
"""

import json
import random
import sys
import os
import subprocess
import re


def get_value(input_value, var_dict):
    """
    get mapping value according to type
    """
    ret_value = 0
    if type(input_value) is int:
        ret_value = input_value
    elif type(input_value) is dict:
        if input_value["type"] == "normal":
            ratio = input_value["var"]["ratio"] if "ratio" in input_value["var"] else 1
            addition = input_value["var"]["addition"] if "addition" in input_value["var"] else 0
            ret_value = var_dict[input_value["var"]["key"]] * ratio + addition
        elif input_value["type"] == "convert":
            if input_value["var"]["key"] not in var_dict:
                ret_value = input_value["default"]
            else:
                ret_value = input_value["kv"][var_dict[input_value["var"]["key"]]]
        else:
            raise Exception("wrong type for range var")
    else:
        raise Exception("wrong type for range params")
    return ret_value


def parse_meta_to_cases(op_meta_path, caseid, dump_path):
    """
    parse meta file, generate case and dump to json file
    """
    # get meta info
    with open(op_meta_path, "r") as f:
        op_meta = json.load(f)
    # parse params
    param_list = []
    var_dict = {}
    for param in op_meta["params"]:
        if "default" in param:
            if random.choice([0, 1, 2, 3]) == 0:
                continue
        param_dict = param_list[param["like"]].copy() if "like" in param else {}
        if "paddle" in param:
            param_dict["name"] = param["paddle"]
        if "tf" in param:
            param_dict["tf-name"] = param["tf"]
        if "like" in param:
            param_list.append(param_dict)
            continue
        pick_type = random.choice(param["type"])
        param_dict["type"] = pick_type
        if pick_type == "variable":
            if param["dim"]["type"] == "range":
                range_list = [0, 0]
                for i in range(2):
                    range_list[i] = get_value(param["dim"]["value"][i], var_dict)
                param_dict["dim"] = random.randrange(*range_list)
            else:
                raise Exception("wrong type for dim type")
            param_dict["data_generator"] = []
            for i in range(param_dict["dim"]):
                if param_dict["dim"] == 4:
                    if i == 0:
                        param_dict["data_generator"].append({"type": "default", "value": random.randint(1, 10)})
                    elif i == 1:
                        param_dict["data_generator"].append({"type": "default", "value": random.randint(1, 3)})
                    else:
                        param_dict["data_generator"].append({"type": "default", "value": random.randint(1, 1000)})
                elif param_dict["dim"] == 3:
                    if i == 0:
                        param_dict["data_generator"].append({"type": "default", "value": random.randint(1, 3)})
                    else:
                        param_dict["data_generator"].append({"type": "default", "value": random.randint(1, 1000)})
                elif param_dict["dim"] >= 5:
                    param_dict["data_generator"].append({"type": "default", "value": random.randint(1, 20)})
                else:
                    param_dict["data_generator"].append({"type": "default", "value": random.randint(1, 1000)})
            param_dict["dtype"] = random.choice(param["dtype"]) if "dtype" in param else "float32"
        elif pick_type in ("tuple", "list"):
            if "size" in param:
                if param["size"]["type"] == "range":
                    range_list = [0, 0]
                    for i in range(2):
                        range_list[i] = get_value(param["size"]["value"][i], var_dict)
                    param_dict["size"] = random.randrange(*range_list)
                elif param["size"]["type"] == "choice":
                    param_dict["size"] = random.choice(param["size"]["option"])
                elif param["size"]["type"] == "default":
                    param_dict["size"] = get_value(param["size"]["value"], var_dict)
                else:
                    raise Exception("wrong type for size type")
            else:
                param_dict["size"] = random.randrange(1, 11)
            param_dict["data_generator"] = []
            for i in range(param_dict["size"]):
                if "element" in param:
                    if param["element"]["type"] == "range":
                        range_list = [0, 0]
                        for i in range(2):
                            range_list[i] = get_value(param["element"]["value"][i], var_dict)
                        param_dict["data_generator"].append({"type": "default",
                                                             "value": random.randrange(*range_list)})
                    else:
                        raise Exception("wrong type for element type")
                else:
                    raise Exception("miss element")
            param_dict["dtype"] = random.choice(param["dtype"]) if "dtype" in param else "float32"
        elif pick_type in ("int", "float"):
            if "element" in param:
                if param["element"]["type"] == "range":
                    range_list = [0, 0]
                    for i in range(2):
                        range_list[i] = get_value(param["element"]["value"][i], var_dict)
                    param_dict["data_generator"] = {"type": "random", "range": [range_list[0], range_list[1] - 1]}
                elif param["element"]["type"] == "default":
                    param_dict["data_generator"] = param["element"]
                elif param["element"]["type"] == "choice":
                    element = random.choice(param["element"]["option"])
                    param_dict["data_generator"] = {"type": "default", "value": element}
                else:
                    raise Exception("wrong type for element type")
            else:
                raise Exception("miss element")
        elif pick_type in ("bool", "string"):
            if "element" in param:
                if param["element"]["type"] == "defualt":
                    param_dict["data_generator"] = param["element"]
                elif param["element"]["type"] == "choice":
                    element = random.choice(param["element"]["option"])
                    param_dict["data_generator"] = {"type": "default", "value": element}
                else:
                    raise Exception("wrong type for element type")
            else:
                raise Exception("miss element")
        elif pick_type == "None":
            pass
        else:
            raise Exception("wrong pick type")
        if "var" in param:
            if param["var"]["value"] in param_dict:
                var_dict[param["var"]["key"]] = param_dict[param["var"]["value"]]
            else:
                var_dict[param["var"]["key"]] = param_dict["data_generator"]["value"]
        param_list.append(param_dict)
    case_key = "{}_{}".format(op_meta["op"], "%03d" % caseid)
    dump = {case_key: {"params": param_list, "cover": 1}}
    # other keys
    for key in op_meta:
        if key == "params":
            continue
        elif key == "level-diff":
            dump[case_key][key] = get_value(op_meta[key], var_dict)
        else:
            dump[case_key][key] = op_meta[key]
    # dump to file
    with open("{}/{}.json".format(dump_path, case_key), "w") as f:
        json.dump(dump, f, indent=4)
    return case_key


def calc_coverage(op_pattern):
    """
    """
    os.system("lcov --capture -d /paddle/build/paddle/fluid/operators/{} -o coverage.info"\
              " --gcov-tool /usr/bin/gcov-4.8 --rc lcov_branch_coverage=1 > /dev/null".format(op_pattern))
    sp = subprocess.Popen("lcov --extract coverage.info '/paddle/paddle/fluid/operators/*'"\
                          " -o coverage.tmp --rc lcov_branch_coverage=1", shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = sp.communicate()
    line_coverage = float(re.search(r"lines.*: (.*)% ", stdout).group(1))
    function_coverage = float(re.search(r"functions.*: (.*)% ", stdout).group(1))
    branch_coverage = float(re.search(r"branches.*: (.*)% ", stdout).group(1))
    return line_coverage, function_coverage, branch_coverage


if __name__ == "__main__":
    """
    pattern = "elementwise"
    meta = "elementwise_add.json"
    line_coverage, function_coverage, branch_coverage = calc_coverage(pattern)
    for j in range(0, 500):
        case_key = parse_meta_to_cases("meta/{}".format(meta), j, "ce/2019_08_23/case")
        os.system("cd bin && python run_ce.py 2019_08_23 {}".format(case_key))
        new_line_coverage, new_function_coverage, new_branch_coverage = calc_coverage(pattern)
        if new_line_coverage > line_coverage or new_function_coverage > function_coverage or new_branch_coverage > branch_coverage:
            with open("ce/2019_08_23/case/{}.json".format(case_key), "r") as fr:
                case_json = json.load(fr)
                case_json[case_key].pop("cover")
                with open("ce/2019_08_23/case_picked/{}.json".format(case_key), "w") as fw:
                    json.dump(case_json, fw, indent=4)
            print line_coverage, function_coverage, branch_coverage, "increase to", new_line_coverage, new_function_coverage, new_branch_coverage
            line_coverage = new_line_coverage
            function_coverage = new_function_coverage
            branch_coverage = new_branch_coverage
    """"""
    for j in range(30):
        parse_meta_to_cases("meta/{}".format("softmax.json"), j, "ce/2019_08_23/case")
    """
    # generate 30 cases for op
    """"""
    meta_list = os.listdir("../meta")
    for meta in sorted(meta_list):
        for j in range(30):
            parse_meta_to_cases("../meta/{}".format(meta), j, "../ce/abnormal/case")
    """"""
