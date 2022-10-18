#!/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import os
import time
import argparse

case_list = {
            "avg_pool1D": "test_avg_pool1D",
            "avg_pool2D": "test_avg_pool2D",
            "avg_pool3D": "test_avg_pool3D",
            "conv1d": "test_conv1d",
            "conv2d": "test_conv2d",
            "conv3d": "test_conv3d",
            "Linear": "test_Linear",
            "relu6": "test_relu6",
            "Sigmoid": "test_Sigmoid",
            "tanh": "test_tanh",
            "abs": "test_abs",
            "addmm": "test_addmm",
            "amin": "test_amin",
            "argmin": "test_argmin",
            "atan": "test_atan",
            "bitwise_and": "test_bitwise_and",
            "bmm": "test_bmm",
            "broadcast_to": "test_broadcast_to",
            "clip": "test_clip",
            "cos": "test_cos",
            "acos": "test_acos",
            "all": "test_all",
            "any": "test_any",
            "argsort": "test_argsort",
            "atan2": "test_atan2",
            "bitwise_not": "test_bitwise_not",
            "broadcast_shape": "test_broadcast_shape",
            "cast": "test_cast",
            "clone": "test_clone",
            "cosh": "test_cosh",
            "add": "test_add",
            "allclose": "test_allclose",
            "arange": "test_arange",
            "asin": "test_asin",
            "bernoulli": "test_bernoulli",
            "bitwise_or": "test_bitwise_or",
            "broadcast_tensors": "test_broadcast_tensors",
            "ceil": "test_ceil",
            "concat": "test_concat",
            "crop": "test_crop",
            "add_n": "test_add_n",
            "amax": "test_amax",
            "argmax": "test_argmax",
            "assign": "test_assign",
            "bincount": "test_bincount",
            "bitwise_xor": "test_bitwise_xor",
            "broadcast_tensors1": "test_broadcast_tensors1",
            "chunk": "test_chunk",
            "conj": "test_conj",
            "cross": "test_cross",
             }

desc = case_list.keys()

case_string = "支持OP如下： \n"
for i in desc:
    case_string = case_string + i + "\n"
case_string = case_string + "输入all全部执行。"
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=case_string)
parser.add_argument("op", help="input a op name")
args = parser.parse_args()


def check(res, op):
    if res == 0:
        print("Test {} pass!".format(op))
        exit(0)
    else:
        print("Test {} failed! Please check log.".format(op))
        exit(1)


def run(op):
    logfile = "test_" + str(int(time.time())) + ".log"
    print(op)

    try:
        if op == "all":
            for v in case_list.values():
                file = v + ".py"
                cmd = "python -m pytest {}".format(file)
                res = os.system("{} >> {}".format(cmd, logfile))
                check(res, v)
            print("log file is {}".format(logfile))
        elif op in case_list.keys():
            file = case_list[op] + ".py"
            print(file)
            cmd = "python -m pytest {}".format(file)
            res = os.system("{} >> {}".format(cmd, logfile))
            check(res, op)
            print("log file is {}".format(logfile))
        else:
            raise FileNotFoundError
    except FileNotFoundError as e:
        print("输入有误！")
        print(case_string)
        exit(1)


if __name__ == '__main__':
    run(args.op)
