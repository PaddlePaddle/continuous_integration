#coding=utf-8
import os
import sys
import yaml
import pymysql
import requests
import smtplib
from email.mime.text import MIMEText
from email.header    import Header
import datetime
import time

import icafe_conf
import mail_conf


db_info = {
    "host": "",
    "port": 1,
    "user": "",
    "password": "",
    "database": "",
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


def mail(sender_addr, receiver_addr, subject, content, proxy):
    msg =  MIMEText(content, 'html', 'UTF-8')
    msg['From'] = sender_addr
    msg['To'] = receiver_addr
    msg['Subject'] = Header(subject, 'UTF-8')

    server = smtplib.SMTP()
    server.connect(proxy)
    try:
        server.sendmail(sender_addr, msg['To'].split(','), msg.as_string())
        print("email send")
    except Exception as e:
        print("发送邮件失败:%s" % (e))
    finally:
        server.quit()


def get_icafe_info(icafe_sequence):
    """
    获取icafe卡片的信息
    """
    get_data = "/{}?".format(icafe_sequence)
    get_data += "&u={}".format(icafe_conf.ICAFE_USERNAME)
    get_data += "&pw={}".format(icafe_conf.ICAFE_PASSWORD)
    res = None
    count = 3
    while count > 0:
        count -= 1 
        content = requests.get(icafe_conf.ICAFE_API_GETCARD_ONLINE+get_data)
        if content.status_code == 200:
            res = content.json()
            break
        time.sleep(5)
    return res


def update_icafe_info(id, icafe_status, icafe_createtime):
    """
    更新数据库
    """
    db = pymysql.connect(host=db_info["host"],
                         port=db_info["port"],
                         user=db_info["user"],
                         password=db_info["password"],
                         database=db_info["database"])
    cursor = db.cursor()

    sql_str = "update tipc_case set icafe_status='{}', icafe_createtime='{}' where id={}".format(icafe_status, icafe_createtime, id)
    cursor.execute(sql_str)

    db.commit()
    cursor.close()
    db.close()


def select_data_day(task_dt):
    """
    查询一天的数据，并统计
    # 查询策略todo，问题：任务时间不在一天、paddle包不确定更新时间，每天跑的paddle包可能不同
    """
    db = pymysql.connect(host=db_info["host"],
                         port=db_info["port"],
                         user=db_info["user"],
                         password=db_info["password"],
                         database=db_info["database"])
    cursor = db.cursor()

    # 查询策略todo，问题：任务时间不在一天、paddle包不确定更新时间，每天跑的paddle包可能不同
    # 查询超时模型
    sql_str = "select * from timeout_model where task_dt='{}' order by chain desc, repo desc".format(task_dt)
    cursor.execute(sql_str)
    res_timeout = cursor.fetchall()

    # 查询未超时模型状态
    #sql_str = "select * from ModelInfo where add_time='{}'".format(task_dt)
    sql_str = "select * from tipc_case where task_dt='{}' order by chain, repo".format(task_dt)
    cursor.execute(sql_str)
    res_case = cursor.fetchall()

    # tongji: return total_res, fail_case, timeout_model, task_env
    total_res = {}
    fail_case = []
    timeout_model = []
    task_env = {}

    for item in res_case:
        _chain = item[5]
        _repo = item[2]
        _model = item[6]
        _case = item[8]
        _status = item[9]
        _icafe_url = item[10]
        _frame_branch = item[12]
        _frame_commit = item[13]
        _paddle_whl = item[14]
        _cuda_version = item[16]
        _cudnn_version = item[17]
        _python_version = item[18]
        _docker_image = item[15]
        if _chain not in total_res.keys():
            total_res[_chain] = {}
        if _repo not in total_res[_chain].keys():
            total_res[_chain][_repo] = {}
        if _model not in total_res[_chain][_repo].keys():
            total_res[_chain][_repo][_model] = {"status": "success"}
        if _status == "failed":
             fail_case.append([_chain, _repo, _model, _case, _icafe_url])
             total_res[_chain][_repo][_model]["status"] = "failed"
        task_env = {
             "frame_branch": _frame_branch,
             "frame_commit": _frame_commit,
             "paddle_whl": _paddle_whl,
             "cuda_version": _cuda_version,
             "cudnn_version": _cudnn_version,
             "python_version": _python_version,
             "docker_image": _docker_image,
             }

    for item in res_timeout:
        _chain = item[5]
        _repo = item[2]
        _model = item[6]
        _frame_branch = item[7]
        _frame_commit = item[8]
        _paddle_whl = item[9]
        _cuda_version = item[11]
        _cudnn_version = item[12]
        _python_version = item[13]
        _docker_image = item[10]
        if [_chain, _repo, _model] in timeout_model:
            continue
        timeout_model.append([_chain, _repo, _model])
        if _chain not in total_res.keys():
            total_res[_chain] = {}
        if _repo not in total_res[_chain].keys():
            total_res[_chain][_repo] = {}
        if _model not in total_res[_chain][_repo].keys():
            total_res[_chain][_repo][_model] = {"status": "timeout"}
        task_env = {
             "frame_branch": _frame_branch,
             "frame_commit": _frame_commit,
             "paddle_whl": _paddle_whl,
             "cuda_version": _cuda_version,
             "cudnn_version": _cudnn_version,
             "python_version": _python_version,
             "docker_image": _docker_image,
             }

    db.commit()
    cursor.close()
    db.close()
    
    tongji_res = {
        "chain_base": {
            "PaddleClas": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleDetection": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleGAN": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleNLP": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleOCR": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleRec": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSeg": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSpeech": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleVideo": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PARL": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PASSL": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PGL": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
        },
        "chain_amp": {
            "PaddleClas": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleDetection": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleGAN": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleNLP": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleOCR": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSeg": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSpeech": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleVideo": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
        },
        "chain_pact_infer_python": {
            "PaddleClas": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleDetection": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleGAN": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleNLP": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleOCR": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSeg": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSpeech": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleVideo": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
        },
        "chain_ptq_infer_python": {
            "PaddleClas": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleDetection": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleGAN": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleNLP": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleOCR": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSeg": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSpeech": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleVideo": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
        },
        "chain_distribution": {
            "PaddleClas": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleDetection": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleGAN": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleNLP": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleOCR": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSeg": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSpeech": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleVideo": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
        },
        "chain_infer_cpp": {
            "PaddleClas": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleDetection": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleGAN": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleNLP": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleOCR": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSeg": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSpeech": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleVideo": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
        },
        "chain_serving_cpp": {
            "PaddleClas": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleDetection": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleGAN": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleNLP": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleOCR": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSeg": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSpeech": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleVideo": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
        },
        "chain_serving_python": {
            "PaddleClas": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleDetection": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleGAN": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleNLP": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleOCR": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSeg": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSpeech": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleVideo": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
        },
        "chain_paddle2onnx": {
            "PaddleClas": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleDetection": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleGAN": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleNLP": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleOCR": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSeg": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleSpeech": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "PaddleVideo": {
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
        },
    }
    for chain, info in total_res.items():
        if chain not in tongji_res.keys():
            tongji_res[chain] = {}
        for repo, models in info.items():
            if repo not in tongji_res[chain].keys():
                tongji_res[chain][repo] = {"success": 0, "failed": 0, "timeout": 0}
            for model, item in models.items():
                if item["status"] == "success":
                    tongji_res[chain][repo]["success"] += 1
                elif item["status"] == "failed":
                    tongji_res[chain][repo]["failed"] +=1
                elif item["status"] == "timeout":
                    tongji_res[chain][repo]["timeout"] +=1
                else:
                    pass
            
    
    return tongji_res, fail_case, timeout_model, task_env


def select_data_week():
    """
    查询一周的数据，并统计
    # 查询策略todo，问题：任务时间不在一天、paddle包不确定更新时间，每天跑的paddle包可能不同
    """
    db = pymysql.connect(host=db_info["host"],
                         port=db_info["port"],
                         user=db_info["user"],
                         password=db_info["password"],
                         database=db_info["database"])
    cursor = db.cursor()

    # 查询本周数据
    sql_str = "select * from tipc_case where DATE_SUB(CURDATE(), INTERVAL 7 DAY) <= date(task_dt) order by task_dt"
    cursor.execute(sql_str)
    res = cursor.fetchall()

    #tongji
    total_res = {
        "chain_list": [],
        "repo_list": [],
        "env_dict": {
            "paddle_branch": [],
            "cuda_version": [],
            "cudnn_version": [],
        },
        "model_list": [],
        "model_fail_list": [],
        "case_total_num": 0,
        "case_fail_num": 0,
        "chain_fail_list": [],
        "repo_fail_list": [],
        "model_total_num": [],
        "model_fail_ratio": 0,
        "case_fail_ratio": 0,
    }
    icafe_res = {
        "total": 0,
        "week_new": 0,
        "fix": 0,
        "case": [],
    }

    total_res["case_total_num"] = len(res)
    for item in res:
        _chain = item[5]
        _repo = item[2]
        _model = item[6]
        _case = item[8]
        _status = item[9]
        _icafe_url = item[10]
        _frame_branch = item[12]
        _frame_commit = item[13]
        _paddle_whl = item[14]
        _cuda_version = item[16]
        _cudnn_version = item[17]
        _python_version = item[18]
        _docker_image = item[15]
        if _chain not in total_res["chain_list"]:
            total_res["chain_list"].append(_chain) 
        if _repo not in total_res["repo_list"]:
            total_res["repo_list"].append(_repo) 
        # chain_repo_model 才能唯一标记一个(次)模型
        _model = _chain + _repo + _model
        if _model not in total_res["model_list"]:
            total_res["model_list"].append(_model) 
        if _status == "failed":
            if _model not in total_res["model_fail_list"]:
                total_res["model_fail_list"].append(_model) 
            total_res["case_fail_num"] += 1
            if _chain not in total_res["chain_fail_list"]:
                total_res["chain_fail_list"].append(_chain)
            if _repo not in total_res["repo_fail_list"]:
                total_res["repo_fail_list"].append(_repo)
        if _frame_branch not in total_res["env_dict"]["paddle_branch"]:
            total_res["env_dict"]["paddle_branch"].append(_frame_branch)
        if _cuda_version not in total_res["env_dict"]["cuda_version"]:
            total_res["env_dict"]["cuda_version"].append(_cuda_version)
        if _cudnn_version not in total_res["env_dict"]["cudnn_version"]:
            total_res["env_dict"]["cudnn_version"].append(_cudnn_version)
    total_res["model_total_num"] =  len(total_res["model_list"])
    total_res["model_fail_ratio"] = len(total_res["model_fail_list"]) / total_res["model_total_num"]
    total_res["case_fail_ratio"] = total_res["case_fail_num"] / total_res["case_total_num"]

    # 查询全部失败case
    sql_str = "select * from tipc_case where status='failed' order by task_dt desc, chain desc, repo desc"
    cursor.execute(sql_str)
    res = cursor.fetchall()

    #icafe_res
    icafe_res["week_new"] = total_res["case_fail_num"]
    icafe_res["total"] = len(res)
    icafe_res["fix"] = 0
    for item in res:
        _id = item[0]
        _icafe_url = item[10]
        _icafe_status = item[19]
        _icafe_createtime = item[20]
        _icafe_rd = item[21]
        _icafe_sequence = item[22]
        _icafe_title = item[23]
        _log = item[11]
        if _icafe_status not in ["关闭", "已关闭", "已关闭_非bug", "测试完成"]:
            # 查询卡片，更新_icafe_status, _icafe_createtime
            # 查询 http://hetu.baidu.com/api/platform/api/show?apiId=540&platformId=1615
            icafe_info = get_icafe_info(_icafe_sequence)
            time.sleep(5)
            if icafe_info == None:
                continue
            if icafe_info["code"] == 200:
                _icafe_status = icafe_info["cards"][0]["status"]
                _icafe_createtime = icafe_info["cards"][0]["createdTime"]
                # 同步更新_icafe_status, _icafe_createtime到数据库
                if (_icafe_status != item[19]) or (_icafe_createtime != item[20]):
                    update_icafe_info(_id, _icafe_status, _icafe_createtime)
            if _icafe_status not in ["关闭", "测试完成"]:
                icafe_res["fix"] += 1
        if _icafe_status not in ["关闭", "已关闭", "已关闭_非bug", "测试完成"]:
            _fixed = "否"
        else:
            _fixed = "是"
        
        icafe_res["case"].append([_icafe_url, _icafe_createtime, _icafe_rd, _fixed, _log])

    return total_res, icafe_res


def create_table_day(tongji_res, fail_case, timeout_model, task_env, task_dt):
    """
    table html
    """
    subject = "[TICP]{}执行结果".format(task_dt)
    content = """
        <html>
        <body>
        <div style="text-align:center;">
        </div>
    """
    #table1
    content += """
        <table border="1" align=center>
        <caption bgcolor="#989898">模型整体运行情况</caption>
        <tr><td>编号</td><td>链条</td><td>套件</td><td>成功</td><td>失败</td><td>超时</td></tr>
    """
    total_success = 0
    total_failed = 0
    total_timeout = 0
    count = 1
    for chain, infos in tongji_res.items():
        for repo, item in infos.items():
            content += """
                <tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
            """.format(count, chain, repo, item["success"], item["failed"], item["timeout"])
            total_success += item["success"]
            total_failed += item["failed"]
            total_timeout += item["timeout"]
            count += 1
    content += """
        <tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
    """.format("总计", "", "", total_success, total_failed, total_timeout)
    content += """
        </table>
        <br>
        目前只有OCR、Seg、Detection、Clas、Video(chain_base)相关链条日志符合标准，其他套件日志未标准化，无法接入统计报表<br>
        日志规范文档：https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/t7n0qKWNJW/QMxJ7wiFu-/aGeuaCrVKznaoM
        <br><br>
    """

    #table2
    if len(fail_case) > 0:
        content += """
            <table border="1" align=center>
            <caption bgcolor="#989898">失败case列表</caption>
            <tr><td>链条</td><td>套件</td><td>模型</td><td>case</td><td>icafe</td></tr>
        """
        for item in fail_case:
            content += """
                <tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>
            """.format(item[0], item[1], item[2], item[3], item[4])
        content += """
            </table>
            <br><br>
        """

    #table3
    if len(timeout_model) > 0:
        content += """
            <table border="1" align=center>
            <caption bgcolor="#989898">超时模型列表</caption>
            <tr><td>链条</td><td>套件</td><td>模型</td></tr>
        """
        for item in timeout_model:
            content += """
                <tr><td>{}</td><td>{}</td><td>{}</td></tr>
            """.format(item[0], item[1], item[2])
        content += """
            </table>
            <br><br>
        """

    #table4
    content += """
        <table border="1" align=center>
        <caption bgcolor="#989898">环境</caption>
    """
    for k, v in task_env.items():
        content += """
            <tr><td>{}</td><td>{}</td></tr>
        """.format(k, v)
    content += """
        </table>
        <br><br>
    """

    content += """
        </table>
        </body>
        </html>
    """

    return subject, content


def create_table_week(total_res, icafe_res):
    """
    table html
    """
    subject = "[TIPC]本周结果汇总"
    content = """
        <html>
        <body>
        <div style="text-align:center;">
        </div>
    """
    #table1
    content += """
        <table border="1" align=center>
        <caption bgcolor="#989898">本周整体结果汇总</caption>
    """
    _chain = "[" + str(len(total_res["chain_list"])) + "个] " + ", ".join(total_res["chain_list"])
    _repo = "[" + str(len(total_res["repo_list"])) + "个] " + ", ".join(total_res["repo_list"])
    _env = "paddle分支: " + ", ".join(total_res["env_dict"]["paddle_branch"]) + "<br>" + \
           "cuda_version: " + ", ".join(total_res["env_dict"]["cuda_version"]) + "<br>" + \
           "cudnn_version: " + ", ".join(total_res["env_dict"]["cudnn_version"])
    _model_num = total_res["model_total_num"]
    _case_num = total_res["case_total_num"]
    _model_fail_ratio =  total_res["model_fail_ratio"]
    _case_fail_ratio =  total_res["case_fail_ratio"]
    _chain_fail = "[" + str(len(total_res["chain_fail_list"])) + "个] " + ", ".join(total_res["chain_fail_list"]) 
    _repo_fail = "[" + str(len(total_res["repo_fail_list"])) + "个] " + ", ".join(total_res["repo_fail_list"]) 
    content += """
        <tr><td>覆盖的链条(及数量)</td><td>{}</td></tr>
        <tr><td>覆盖的套件(及数量)</td><td>{}</td></tr>
        <tr><td>覆盖的环境</td><td>{}</td></tr>
        <tr><td>累计例行模型总次</td><td>{}</td></tr>
        <tr><td>累计例行case总次</td><td>{}</td></tr>
        <tr><td>模型失败率</td><td>{}</td></tr>
        <tr><td>case失败率</td><td>{}</td></tr>
        <tr><td>失败case覆盖的链条</td><td>{}</td></tr>
        <tr><td>失败case覆盖的套件</td><td>{}</td></tr>
    """.format(_chain, _repo, _env, _model_num, _case_num, _model_fail_ratio, _case_fail_ratio, _chain_fail, _repo_fail)
    content += """
        </table>
        <br><br>
    """

    #table2
    content += """
        <table border="1" align=center>
        <caption bgcolor="#989898">失败case列表</caption>
    """
    #content += """
    #    <caption bgcolor="#989898">总计: {}个</caption>
    #    <caption bgcolor="#989898">本周新增: {}个</caption>
    #    <caption bgcolor="#989898">待修复: {}个</caption>
    #""".format(icafe_res["total"], icafe_res["week_new"], icafe_res["fix"])
    content += """
       <tr><td>总计: {}个<br>本周新增: {}个<br>待修复: {}个<br></td></tr>
    """.format(icafe_res["total"], icafe_res["week_new"], icafe_res["fix"])
    content += """
        <tr><td>icafe地址</td><td>创建时间</td><td>rd负责人</td><td>是否修复</td><td>原始log</td></tr>
    """
    for item in icafe_res["case"]:
        content += """
            <tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr> 
        """.format(item[0], item[1], item[2], item[3], item[4])
    content += """
        </table>
        <br><br>
    """

    content += """
        </table>
        </body>
        </html>
    """

    return subject, content


def report_day(sender, reciver, proxy):
    """
    天级报告
    """
    get_db_info()
    task_dt = datetime.date.today()
    oneday=datetime.timedelta(days=1)
    task_dt = task_dt - oneday
    total_res, fail_case, timeout_model, task_env = select_data_day(task_dt)
    subject, content = create_table_day(total_res, fail_case, timeout_model, task_env, task_dt)
    mail(sender, reciver, subject, content, proxy)


def report_week(sender, reciver, proxy):
    """
    周级报告
    """
    get_db_info()
    total_res, icafe_res = select_data_week()
    subject, content = create_table_week(total_res, icafe_res)
    mail(sender, reciver, subject, content, proxy)


def run(mode):
    sender = mail_conf.SENDER
    reciver = mail_conf.RECIVER
    proxy = mail_conf.PROXY
    if mode == "day":
        report_day(sender, reciver, proxy)
    if mode == "week":
        report_week(sender, reciver, proxy)

if __name__ == "__main__":
    #report_day()
    mode=sys.argv[1] 
    run(mode)
