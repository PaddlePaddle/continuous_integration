#!/bin/env python
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
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        universal_newlines=True)
    out, err = process.communicate()
    return out, process.returncode


def get_cmd(parameter_dict):
    p = parameter_dict
    models_zoo = {
        'mobilenetv1': 'bash mobilenetv1.sh',
        'mobilenetv2': 'bash mobilenetv2.sh',
        'resnet18': 'bash resnet18.sh',
        'resnet34': 'bash resnet34.sh',
        'resnet50': 'bash resnet50.sh',
        'resnet101': 'bash resnet101.sh',
        'vgg11': 'bash vgg11.sh',
        'vgg13': 'bash vgg13.sh',
        'vgg16': 'bash vgg16.sh',
        'vgg19': 'bash vgg19.sh'
    }
    return models_zoo[p['model_name']]


## HERE
@app.route("/tool-4")
def run():

    parameter_dict = request.get_json()
    cmd = get_cmd(parameter_dict)
    print(parameter_dict, cmd)

    out, ret = run_cmd(cmd)
    if ret != 0:
        result = {"status": 500, "msg": out, "result": "FAIL"}
    else:
        result = {"status": 200, "msg": out, "result": "PASS"}
    return json.dumps(result)


if __name__ == '__main__':
    ## HERE
    app.run(host="0.0.0.0", port=8704, debug=False)
