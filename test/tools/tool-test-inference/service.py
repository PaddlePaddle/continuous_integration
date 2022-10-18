# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import os
import time
import argparse
import requests
import json
import time
import datetime
from flask import Flask
from flask_restful import Resource, Api
from flask import request


app = Flask(__name__)
api = Api(app)


def run_cmd(cmd):
    import subprocess
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, universal_newlines=True)
    out, err = process.communicate()
    return out, process.returncode


def get_cmd(parameter_dict):
    model_dict = {
        "resnet50": "test_resnet50",
        "bert": "test_bert",
        "ernie": "test_ernie",
    }
    model_name = parameter_dict["model_name"]
    device = parameter_dict["device"]
    blaslib = parameter_dict["blaslib"]
    precision = parameter_dict["precision"]
    if model_name in model_dict.keys():
        if device == "cpu" and blaslib == "mkldnn":
            return f"python -m pytest -sv {model_dict[model_name]}.py -k 'mkldnn'"
        elif device == "gpu" and blaslib == "tensorrt" and precision == "fp32":
            return f"python -m pytest -sv {model_dict[model_name]}.py -k 'trt_fp32'"
        elif device == "gpu" and blaslib == "tensorrt" and precision == "fp16":
            return f"python -m pytest -sv {model_dict[model_name]}.py -k 'trt_fp16'"
        else:
            return f"python -m pytest -sv {model_dict[model_name]}.py -k 'gpu_bz1'"
    else:
        return None


@app.route("/tool-6")
def run():
    parameter_dict = request.get_json()
    cmd = get_cmd(parameter_dict)
    print(parameter_dict, cmd)

    if cmd is None:
        result = {"status": 500, "msg": "wrong model_name", "result": "FAIL"}
        return json.dumps(result)
    out, ret = run_cmd(cmd)
    if ret != 0:
        result = {"status": 500, "msg": out, "result": "FAIL"}
    else:
        result = {"status": 200, "msg": "", "result": "PASS"}
    return json.dumps(result)


if __name__ == '__main__':
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib64:{}".format(ld_library_path)
    app.run(host="0.0.0.0", port=8706, debug=False)
