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
        "AlexNet": "AlexNet",
        "ResNet50": "ResNet50",
        "DarkNet53": "DarkNet53",
        "MobileNetV1": "MobileNetV1",
        "EfficientNetB0": "EfficientNetB0",
        "MobileNetV2": "MobileNetV2",
    }
    model_name = parameter_dict["model_name"]
    model_path = f"./PaddleClas/infer_static/{model_name}/__model__"
    params_path = f"./PaddleClas/infer_static/{model_name}/params"
    if model_name in model_dict.keys():
        return f"python clas_benchmark.py --model_name={model_name} " \
               f"--model_path={model_path} " \
               f"--params_path={params_path} " \
               f"--repeats=1000 " \
               f"--batch_size=1 " \
               f"--use_gpu"
    else:
        return None


@app.route("/tool-11")
def run():
    parameter_dict = request.get_json()
    cmd = get_cmd(parameter_dict)
    print(parameter_dict, cmd)

    if cmd is None:
        result = {"status": 500, "msg": "wrong model_name", "result": "FAIL"}
        return json.dumps(result)
    out, ret = run_cmd(cmd)
    print("status code", ret)
    print("result", out)
    if ret != 0:
        result = {"status": 500, "msg": out, "result": "FAIL"}
    else:
        output_dict = {}
        for line in out.split("\n"):
            line_list = line.split(" ")
            if "QPS:" in line_list and "latency(ms):" in line_list:
                pos_buf = line_list.index("QPS:")
                output_dict["avg_latency_ms"] = line_list[pos_buf - 1].split(',')[0]
                output_dict["qps"] = line_list[-1].strip()
        result = {"status": 200, "msg": output_dict, "result": "PASS"}
    return json.dumps(result)


if __name__ == '__main__':
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib64:{}".format(ld_library_path)
    if not os.path.exists("PaddleClas"):
        print("download model...")
        cmd = "wget --no-proxy -q https://sys-p0.bj.bcebos.com/Paddle-UnitTest-Model/PaddleClas.tgz --no-check-certificate && tar -zxf PaddleClas.tgz;"
        os.system(cmd)
    app.run(host="0.0.0.0", port=8711, debug=False)
