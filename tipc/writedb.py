"""
包含上传log、创建icafe、结果入库
"""
#coding=utf-8

import sys
import pymysql
import yaml
import copy
import subprocess
import os
import requests
import json
import re

import icafe_conf

task_env = {
    "task_dt": sys.argv[1],
    "repo": sys.argv[2],
    "repo_branch": sys.argv[3],
    "repo_commit": sys.argv[4],
    "chain": sys.argv[5],
    "paddle_whl": sys.argv[6],
    "frame_branch": sys.argv[7],
    "frame_commit": sys.argv[8],
    "docker_image": sys.argv[9],
    "cuda_version": sys.argv[10],
    "cudnn_version": sys.argv[11],
    "python_version": sys.argv[12],
}


db_info = {
    "host": "",
    "port": 1,
    "user": "",
    "password": "",
    "database": "",
}


res = {
    "models_status": {},
    "total_num": 0,
    "timeout_num": 0,
    "success_num": 0,
    "failed_num": 0,
    "failed_models": [],
    "success_models": [],
    "timeout_models": [],
    "model_func": {},
    "failed_cases_num": 0,
    "success_cases_num": 0,
}


def get_db_info():
    """
    """
    with open("db_info.yaml", "r") as fin:
        file_date = yaml.load(fin.read(), Loader=yaml.Loader)
        db_info["host"] = file_date["host"]
        db_info["port"] = int(file_date["port"])
        db_info["user"] = file_date["user"]
        db_info["password"] = file_date["password"]
        db_info["database"] = file_date["database"]


def get_model_info():
    """
    """
    with open("full_chain_list_all", "r") as fin:
        lines = fin.readlines()
        res["total_num"] = len(lines)
    with open("TIMEOUT", "r") as fin:
        lines = fin.readlines()
        res["timeout_num"] = len(lines)
        for line in lines:
            tmp = line.split(" ")
            model_name = tmp[0]
            res["timeout_models"].append(model_name)
    with open("RESULT", "r") as fin:
        lines = fin.readlines()
        for line in lines:
            tmp = line.split(" - ")
            model_name = tmp[1].strip()
            case = tmp[2]
            log_path = tmp[3].split(" ")[0].strip() ## 计划在RESULT中加上日志地址tmp[3]
            icafe_url = ""
            icafe_createtime = None
            icafe_sequence = 0
            icafe_title = ""
            icafe_status = ""
            icafe_rd = ""
            if model_name in res["timeout_models"]:
                continue
            stage = ""
            if ("train.py --test-only" in case) or ("main.py --test" in case):
                stage = "eval"
            elif ("train.py" in case) or ("main.py --validat" in case) or ("train_copy.py" in case):
                stage = "train"
            elif ("export_model.py" in case) or ("export.py" in case) or ("to_static.py" in case):
                stage = "dygraph2static"
            elif ("infer.py" in case) or ("predict_det.py" in case):
                stage = "inference"
            else:
                stage = "inference"
            if "successfully" in tmp[0]:
                tag = "success"
                res["success_cases_num"] += 1
                log_path = ""
            else:
                tag = "failed"
                res["failed_cases_num"] += 1
                # upload log to bos
                log_path = upload_log(model_name, log_path, task_env["chain"]) # 上传log到bos
                # create icafe
                icafe_params = {"title": "", "detail": "", "repo": "", "rd": ""}
                icafe_params["title"] = "[auto][tipc][{}]{} {} {} {} 失败".format(task_env["task_dt"], task_env["repo"], task_env["chain"], model_name, stage)
                icafe_params["detail"] = "套件：{}\r\n链条：{}\r\n模型：{}\r\ncase：{}\r\n日志：{}".format(task_env["repo"], task_env["chain"], model_name, case, log_path)
                icafe_params["repo"] = task_env["repo"]
                icafe_params["rd"] = icafe_conf.RD[task_env["repo"]]
                icafe_url, icafe_createtime, icafe_sequence, icafe_title, icafe_status = create_icafe(icafe_params)
                icafe_rd = icafe_params["rd"]
            if model_name not in res["models_status"].keys():
                res["models_status"].setdefault(model_name, [])
            res["models_status"][model_name].append({"status": tag, "case": case, "stage": stage, "icafe_url": icafe_url, "icafe_status": icafe_status, "icafe_createtime": icafe_createtime, "icafe_rd": icafe_rd, "icafe_sequence": icafe_sequence, "icafe_title": icafe_title, "log": log_path})


def upload_log(model_name, log_path, chain):
    """
    日志上传地址：https://paddle-qa.bj.bcebos.com/fullchain_ce_test/${time_stamp}^${REPO}^${CHAIN}^${model_name}^${paddle_commit}^${repo_commit}^${log_path}
    """
    cmd = "bash upload_log.sh {} {} {} {} {}".format(model_name, log_path, chain, task_env["frame_commit"], task_env["repo_commit"])
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, universal_newlines=True)
    out, err = process.communicate()
    log_path = out.split("\n")[-2]
    if "paddle-qa.bj.bcebos.com" not in log_path:
        log_path = ""
    return log_path


def create_icafe(icafe_params):
    """
    失败case创建icafe bug卡片
    icafe_params:
        title
        detail
        repo
        rd
    """
    icafe_dict = {'username': icafe_conf.ICAFE_USERNAME, 'password': icafe_conf.ICAFE_PASSWORD, 'issues': []}
    item_dict = {
         'title': icafe_params['title'],
         'detail': icafe_params['detail'],
         'type': 'Bug',
         'parent': '55012',
         'fields': {
            '流程状态': '新建',
            'auto_tag': 'auto_bug',
            'repo': icafe_params['repo'],
            '需求来源': 'QA团队',
            '负责人': 'zhengya01',
            'QA负责人': 'zhengya01',
            'RD负责人': icafe_params['rd'],
            '负责人所属团队': 'QA团队',
            'bug发现方式': '模型套件',
            '优先级': 'P1-严重问题 High',
        }
    }
    icafe_dict['issues'].append(item_dict)

    r = requests.post(icafe_conf.ICAFE_API_NEWCARD_ONLINE, data=json.dumps(icafe_dict))
    print(r)
    res_json = r.json()
    icafe_url = ""
    icafe_createtime = None
    icafe_sequence = ""
    icafe_title = ""
    icafe_status = ""
    if res_json["status"] == 200:
        icafe_url = res_json["issues"][0]["url"]
        icafe_sequence = res_json["issues"][0]["sequence"]
        icafe_title = res_json["issues"][0]["title"]
        icafe_status = "新建"
    else:
        print("failed")
        print(res_json)

    return icafe_url, icafe_createtime, icafe_sequence, icafe_title, icafe_status


def write():
    """
    """
    db = pymysql.connect(host=db_info["host"],
                         port=db_info["port"],
                         user=db_info["user"],
                         password=db_info["password"],
                         database=db_info["database"])
    cursor = db.cursor()

    # cases
    sql_str = "insert into tipc_case \
                        (task_dt, \
                         repo, repo_branch, repo_commit, \
                         chain, \
                         paddle_whl, frame_branch, frame_commit, \
                         docker_image, cuda_version, cudnn_version, python_version, \
                         model, stage, cmd, status, icafe_url, log, icafe_status, icafe_createtime, icafe_rd, icafe_sequence, icafe_title) \
                        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    val = []
    for model, infos in res["models_status"].items():
        for item in infos:
            val.append((task_env["task_dt"],
                       task_env["repo"], task_env["repo_branch"], task_env["repo_commit"],
                       task_env["chain"],
                       task_env["paddle_whl"], task_env["frame_branch"], task_env["frame_commit"],
                       task_env["docker_image"], task_env["cuda_version"], task_env["cudnn_version"], task_env["python_version"],
                       model, item["stage"], item["case"], item["status"], item["icafe_url"], item["log"], item["icafe_status"], item["icafe_createtime"], item["icafe_rd"], item["icafe_sequence"], item["icafe_title"]))
    cursor.executemany(sql_str, val)
    db.commit()

    # Timeout models
    sql_str = "insert into timeout_model \
                        (task_dt, \
                         repo, repo_branch, repo_commit, \
                         chain, \
                         paddle_whl, frame_branch, frame_commit, \
                         docker_image, cuda_version, cudnn_version, python_version, \
                         model) \
                        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    val = []
    for model in res["timeout_models"]:
         val.append((task_env["task_dt"],
                       task_env["repo"], task_env["repo_branch"], task_env["repo_commit"],
                       task_env["chain"],
                       task_env["paddle_whl"], task_env["frame_branch"], task_env["frame_commit"],
                       task_env["docker_image"], task_env["cuda_version"], task_env["cudnn_version"], task_env["python_version"],
                       model))
    cursor.executemany(sql_str, val)
    db.commit()

    db.close()


def run():
    """
    """
    get_model_info()
    get_db_info()
    write()


if __name__ == "__main__":
    run()
