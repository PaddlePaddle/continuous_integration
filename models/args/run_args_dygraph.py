#!/usr/bin/env python
"""
"""

#system lib
from __future__ import division
import copy
import os
import sys

#self lib
import test_args


def dict2argstr(args_dict):
    """
    transfor dict to str
    params:
        args_dict: 
    return:
        str
    """
    arg_str = ""
    for arg, value in args_dict.items():
        if value is not None:
            arg_str += " --{} {}".format(str(arg), str(value))
    return arg_str


def run(run_py, func):
    """
    excute run_py with diff args
    run_py : the excutable py file
    func : [train|infer|...]
    """
    args = eval("test_args.{}".format(func))
    print(args)

    res = {}

    default_args = {}
    for arg, value in args.items():
        default_args[arg] = value[0]

    current_args = dict2argstr(default_args) 
    cmd = "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; \
      python -m paddle.distributed.launch  --selected_gpus=0,1,2,3  {} {}".format(
      run_py, current_args)
    status = os.system(cmd)
    if status != 0:
        res[cmd] = "FAIL"
    else:
        res[cmd] = "SUCCESS"
    cmd = "rm -rf checkpoints"
    os.system(cmd)

    for arg, value in args.items():
        if len(value) <= 1:
            continue
        current_args_dict = copy.deepcopy(default_args)
        for item in value[1:]:
            current_args_dict[arg] = item
            current_args = dict2argstr(current_args_dict)
            cmd = "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; \
              python -m paddle.distributed.launch  --selected_gpus=0,1,2,3  {} {}".format(
              run_py, current_args)
            status = os.system(cmd)
            if status != 0:
                res[cmd] = "FAIL"
            else:
                res[cmd] = "SUCCESS"
            cmd = "rm -rf checkpoints"
            os.system(cmd)

    total_num = len(res)
    fail_num = 0
    for cmd, status in res.items():
        if status == "FAIL":
            fail_num += 1
    print("-" * 30)
    print("Failure Rate: {} / {}".format(str(fail_num), str(total_num)))
    print("-" * 30)
    print("Detail:")
    for cmd, status in res.items():
        print("{} : {}".format(status, cmd))


if __name__ == "__main__":
    run_py = sys.argv[1]
    func = sys.argv[2]
    run(run_py, func)
