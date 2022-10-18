# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tool Interface
"""

import requests
import json
import time
import datetime
from flask import Flask
from flask_restful import Resource, Api
from flask import request
import argparse
import traceback2 as traceback
import os
import logging
import logging.handlers

app = Flask(__name__)
api = Api(app)
API_PORT = 8300


def init_log(
        log_path,
        level=logging.INFO,
        when="D",
        backup=7,
        format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s",
        datefmt="%m-%d %H:%M:%S"):
    """init_log - initialize log module

    Args:
    log_path - Log file path prefix.
    Log data will go to two files: log_path.log and log_path.log.wf
    Any non-exist parent directories will be created automatically
    level - msg above the level will be displayed
    DEBUG < INFO < WARNING < ERROR < CRITICAL
    the default value is logging.INFO
    when - how to split the log file by time interval
    'S' : Seconds
    'M' : Minutes
    'H' : Hours
    'D' : Days
    'W' : Week day
    default value: 'D'
    format - format of the log
    default format:
    %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s
    INFO: 12-09 18:02:42: log.py:40 * 139814749787872 HELLO WORLD
    backup - how many backup file to keep
    default value: 7

    Raises:
    OSError: fail to create log directories
    IOError: fail to open log file
    """
    formatter = logging.Formatter(format, datefmt)
    logger = logging.getLogger()
    logger.setLevel(level)

    dir = os.path.dirname(log_path)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    handler = logging.handlers.TimedRotatingFileHandler(
        log_path + ".log", when=when, backupCount=backup)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.handlers.TimedRotatingFileHandler(
        log_path + ".log.wf", when=when, backupCount=backup)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--callback_addr")
    args = parser.parse_args()
    return args


def callback(result, standard_no):
    """
    callback
    """
    ret = {"standardNo": standard_no, "channelName": "baidu", "result": True}
    if not result.get("status") == 200:
        ret["result"] = False
    logging.info(ret)
    args = parse_args()
    if args.callback_addr:
        try:
            request_url = 'http://{}/doaction/callback/baidu/evaluation/results'.format(
                args.callback_addr)
            r = requests.post(request_url, \
                headers={'Content-Type':'application/json', 'Accept':'application/json', 'charset':'UTF-8'}, \
                data=ret)
            logging.info(r)
        except Exception as e:
            logging.info(e)
            logging.info(traceback.format_exc())
    return result


@app.route("/tool-<int:toolid>")
def service(toolid):
    """
    Args:
        toolid: Tool ID.
    Returns:
        Http response(json).
    """
    try:
        framework_port_dict = {'paddle': 8700, }
        framework_name = 'paddle'
        port = str(framework_port_dict[framework_name] + toolid)
        request_url = request.url.replace(str(API_PORT), port)
        logging.info(request_url)
        r = requests.get(request_url, \
            headers={'Content-Type':'application/json', 'Accept':'application/json', 'charset':'UTF-8'}, \
            data=json.dumps(request.get_json()))
        try:
            response_json = json.loads(r.text)
            if not response_json.get("status") == 200:
                msg = response_json.get("msg", "")
                logging.info("Fail, details: %s " % msg)
            return callback(response_json, request.get_json().get("standardNo"))
        except:
            return callback({
                "status": 500,
                "msg": r.text,
                "result": "FAIL"
            }, request.get_json().get("standardNo"))
    except Exception as e:
        logging.info(e)
        msg = str(traceback.format_exc())
        logging.info(msg)
        return callback({
            "status": 500,
            "msg": msg,
            "result": "FAIL"
        }, request.get_json().get("standardNo"))


if __name__ == "__main__":
    init_log("./log/api.log")
    app.run(host="0.0.0.0", port=API_PORT, debug=True)
