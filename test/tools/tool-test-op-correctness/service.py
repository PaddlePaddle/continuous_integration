#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import os
import json
from flask import Flask
from flask_restful import Api
from flask import request

app = Flask(__name__)
api = Api(app)


@app.route("/tool-7")
# def run(op):
def run():
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
    result = {
        "status": 200,
        "msg": "",
        "result": "PASS",
    }
    # get request获取并解析参数
    req = request.get_json()
    # check whether param is illegal
    if req.get("op_name") not in case_list.keys():
        result['status'] = 500
        result['msg'] = "op_name paramter error"
        result['result'] = "FAIL"
        return result

    # your case cmd 执行脚本命令
    # cases_dir相对路径从根目录开始
    cases_dir = "./cases"
    res = os.system("cd {} && python run.py {}".format(cases_dir,
                                                       req.get("op_name")))

    parameter_dict = request.get_json()
    op = parameter_dict.get("op_name")
    # op = request.args["op_name"]
    # Finish
    case_result = True if res == 0 else False

    # 返回值 自定义报错信息
    # good return
    if case_result:
        pass
    # bad return
    else:
        result['status'] = 400
        result['msg'] = "test failed"
        result['result'] = "FAIL"
    return json.dumps(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8707, debug=True)
