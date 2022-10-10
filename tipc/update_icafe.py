#coding=utf-8
import os
import sys
import yaml
import pymysql
import requests
import datetime
import time
import json

import icafe_conf


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


def select_icafe_sequence(key_str):
    """
    根据给定的关键字符串key_str，从数据库中查询icafe_title中包含key_str的icafe_sequence
    """
    db = pymysql.connect(host=db_info["host"],
                         port=db_info["port"],
                         user=db_info["user"],
                         password=db_info["password"],
                         database=db_info["database"])
    cursor = db.cursor()

    # 查询策略todo，问题：任务时间不在一天、paddle包不确定更新时间，每天跑的paddle包可能不同
    # 查询超时模型
    sql_str = "select icafe_sequence from tipc_case where icafe_title like '%{}%'".format(key_str)
    print(sql_str)
    cursor.execute(sql_str)
    res = cursor.fetchall()
    icafe_sequences = []
    for item in res:
        icafe_sequences.append(item[0])
    return icafe_sequences


def update_icafe(icafe_sequence, status, type_1, type_2, type_q):
    """
    根据给定条件，修改icafe_sequence对应的icafe卡片的相关字段
    """
    _data = "/{}?".format(icafe_sequence)
    _data += "u={}".format(icafe_conf.ICAFE_USERNAME)
    _data += "&pw={}".format(icafe_conf.ICAFE_PASSWORD)
 
    if status == "已关闭_非Bug": 
        params = {"fields": 
                      [
                      "流程状态={}".format(status) 
                      ]
                 }
    else:
        params = {"fields": 
                      [
                      "流程状态={}".format(status), 
                      "所属方向={}".format(type_1),
                      "所属方向-细分={}".format(type_2),
                      "问题细分类={}".format(type_q)
                      ]
                 }
    print(icafe_conf.ICAFE_API_UPDATECARD_ONLINE+_data)
    print(params)
    res = requests.post(icafe_conf.ICAFE_API_UPDATECARD_ONLINE+_data, data=params)
    print(res)


def run():
    """
    """
    key_str = sys.argv[1]
    status = sys.argv[2]
    type_1 = ""
    type_2 = ""
    type_q = ""
    if (status != "已关闭_非Bug") and len(sys.argv) < 6:
        print("argv error")
        exit(-1)
    if len(sys.argv) > 5:
        type_1 = sys.argv[3]
        type_2 = sys.argv[4]
        type_q = sys.argv[5]
    get_db_info()
    icafe_sequences = select_icafe_sequence(key_str)
    for icafe_sequence in icafe_sequences:
        update_icafe(icafe_sequence, status, type_1, type_2, type_q)
        time.sleep(2)


if __name__ == "__main__":
    run()
