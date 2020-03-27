#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################

"""
File: web_main.py
Author: xuguangyao(xuguangyao01@baidu.com)
Date: 2019/05/15 16:45:28
"""

from flask import Flask, request, send_from_directory, make_response, jsonify
import os
import json


app = Flask(__name__, static_url_path='/')

@app.route('/')
def index():
    """
    main.html
    """
    return send_from_directory('views', 'main.html')


@app.route('/ce')
def ce_page():
    """
    ce_page.html
    """
    return send_from_directory('views', 'ce_page.html')


@app.route('/node_modules/<path:path>')
def static_lib(path):
    """
    static resource
    """
    return send_from_directory('node_modules', path)


@app.route('/run', methods=['POST'])
def run():
    """
    run op suite
    """
    with open(os.path.join("case", request.form["op"] + ".json"), "r") as fin:
        suite = json.load(fin)
        for case_name in suite:
            print "BEGIN CASE: {}".format(case_name)
            cmd = "cd bin && export CUDA_VISIBLE_DEVICES=0 && python paddle_op_test.py"\
                  " '{0}' {1} fast ../result > ../log/{1}.fast 2>&1".format(json.dumps(suite[case_name]), case_name)
            os.system(cmd)
        print ""
    return jsonify({"status": 0, "result": combine_dict(request.form["op"], _id=request.form["id"])})


@app.route('/save', methods=['POST'])
def save():
    """
    save
    """
    print request.form["detail"]
    with open(os.path.join("case", request.form["op"] + ".json"), "w") as fout:
        fout.write(request.form["detail"])
    return jsonify({"status": 0})


def combine_dict(_op, _id=None, prefix="", is_detail=False):
    """
    combine dict
    """
    with open(os.path.join(prefix + "case", _op + ".json"), "r") as fsuite:
        suite_content = fsuite.read()
    suite_json = json.loads(suite_content)
    if is_detail:
        _dict = {
            "detail": suite_content,
            "console": ""
        }
        for case in suite_json:
            if os.path.exists(os.path.join(prefix + "log", case + ".fast")):
                with open(os.path.join(prefix + "log", case + ".fast"), "r") as flog:
                    _dict["console"] += "BEGIN CASE: {}\n".format(case)
                    _dict["console"] += flog.read()
                    _dict["console"] += "END CASE: {}\n".format(case)
                    _dict["console"] += "\n\n"
        return _dict

    _dict = {
        "id": _id,
        "op": _op,
        #"detail": suite_content,
        #"console": "",
    }
    _func_dict = {}
    _perf_dict = {}
    _stab_dict = {}
    _mem_dict = {}
    for case in suite_json:
        if os.path.exists(os.path.join(prefix + "result", case)):
            with open(os.path.join(prefix + "result", case), "r") as fresult:
                result_dict = json.load(fresult)
                _func_dict[case] = result_dict["function"]
                _perf_dict[case] = result_dict["performance"]
                _stab_dict[case] = result_dict["stablity"]
                _mem_dict[case] = result_dict.get("memory", "-")
        """
        if os.path.exists(os.path.join(prefix + "log", case + ".fast")):
            with open(os.path.join(prefix + "log", case + ".fast"), "r") as flog:
                _dict["console"] += "BEGIN CASE: {}\n".format(case)
                _dict["console"] += flog.read()
                _dict["console"] += "END CASE: {}\n".format(case)
                _dict["console"] += "\n\n"
        """
    if prefix == "":
        _dict["function"] = "<br>".join([case + ":\t" + _func_dict[case] for case in _func_dict])
        _dict["performance"] = "<br>".join([case + ":\t" + _perf_dict[case] for case in _perf_dict])
        _dict["stablity"] = "<br>".join([case + ":\t" + _stab_dict[case] for case in _stab_dict])
        _dict["memory"] = "<br>".join([case + ":\t" + _mem_dict[case] for case in _mem_dict])
    else:
        _dict["function"] = "<br>".join([_func_dict[case] for case in _func_dict])
        _dict["performance"] = "<br>".join([_perf_dict[case] for case in _perf_dict])
        _dict["stablity"] = "<br>".join([_stab_dict[case] for case in _stab_dict])
        _dict["memory"] = "<br>".join([_mem_dict[case] for case in _mem_dict])
    return _dict


@app.route('/get/<stage>/table/info', methods=['POST'])
def get_table_info(stage):
    """
    get table info
    """
    prefix = "ce/{}/".format(request.form["batch_id"]) if stage == "ce" else ""
    suite_file_list = os.listdir(prefix + "case")
    table_lines = []
    for _id, suite_file in enumerate(sorted(suite_file_list)):
        _op = suite_file[:-5]
        table_lines.append(combine_dict(_op, _id, prefix))
    return jsonify({"status": 0, "result": table_lines})


@app.route('/get/<stage>/detail', methods=['POST'])
def get_detail(stage):
    """
    get detail
    """
    prefix = "ce/{}/".format(request.form["batch_id"]) if stage == "ce" else ""
    _op = request.form["case"]
    return jsonify({"status": 0, "result": combine_dict(_op, prefix=prefix, is_detail=True)})


@app.route('/get/ce/list', methods=['POST'])
def get_ce_list():
    """
    get ce list
    """
    ce_list = os.listdir("./ce")
    return jsonify({"status": 0, "result": ce_list})


if __name__ == '__main__':
    app.secret_key = os.urandom(24)
    app.run(host='0.0.0.0', port=8012, debug=True, threaded=True)
