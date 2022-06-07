"""
@Desc: get weekly data
@Author: guolixin
@Date: 20220422 15:00
"""
import os
import sys
import json
import logging
import time
import smtplib
import re
import argparse
from pathlib  import Path
from datetime import datetime

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(base_path)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myframework.settings')
import django
django.setup()

from django.conf import settings
from myapp import models
from django.db.models import Q
from EmailTemplate import EmailTemplate
from collections import OrderedDict
from email.mime.text import MIMEText
from email.header    import Header
from django.db import connections
from save_check import send_mail

WEEK_TOTAL_TABLE_HEADER = ["模型类型", "运行环境", "任务数量",
                     "相对上周下降5%以上个数", "相对上周上升5%以上个数", 
                     "相对上周小于5%个数",
                     "标准值上升5%以上个数", "标准值上升低于5%个数",
                     "上周运行失败个数", "本周运行失败个数"]
WEEK_OTHER_HEADER_HOLDER = ["模型类型", "运行环境", "pytorch", "tensorflow", "oneflow", "mxnet", "mindspore"]
WEEK_RESULT_TABLE_HEADER = ["模型", "运行环境", "本周结果", "上周结果", "本周相对上周波幅",
                     "本周标准值", "上周标准值", "本周标准值相对上周波幅"]
WEEK_DICT_INDEX = {1: "total", 2: "other", 3: "dy_speed", 4: "st_speed", 5:"dy2st_speed"}
WEEK_TOTAL_NUM = {}
WEEK_WAVE_THRESHOLD = 0.05
BKCOLOR = {"N1C1":"Cornsilk", "N1C8":"MistyRose", "N4C32":"LightCyan"}
FRAMES = ["pytorch", "tensorflow", "oneflow", "mxnet", "mindspore"]

def initial_data():
    """
    initial data
    """
    html_results = OrderedDict()
    for k in WEEK_DICT_INDEX.values():
        html_results[k] = {}
        if k == 'total':
            html_results[k]["header"] = WEEK_TOTAL_TABLE_HEADER
        elif k == "other":
            html_results[k]["header"] = WEEK_OTHER_HEADER_HOLDER
        else:
            html_results[k]["header"] = WEEK_RESULT_TABLE_HEADER
        html_results[k]["data"] = []
    return html_results


def construct_dict(model_type, device_num):
    """
    construct dict 
    """
    if model_type not in WEEK_TOTAL_NUM:
        WEEK_TOTAL_NUM[model_type] = {}
    if device_num not in WEEK_TOTAL_NUM[model_type]:
        WEEK_TOTAL_NUM[model_type][device_num] = {}
    if "total" not in WEEK_TOTAL_NUM[model_type][device_num]:
        WEEK_TOTAL_NUM[model_type][device_num]["total"] = 0
    if "pos_lt_st" not in WEEK_TOTAL_NUM[model_type][device_num]:
        WEEK_TOTAL_NUM[model_type][device_num]["pos_lt_st"] = 0
    if "neg_gt_st" not in WEEK_TOTAL_NUM[model_type][device_num]:
        WEEK_TOTAL_NUM[model_type][device_num]["neg_gt_st"] = 0
    if "between_st" not in WEEK_TOTAL_NUM[model_type][device_num]:
        WEEK_TOTAL_NUM[model_type][device_num]["between_st"] = 0
    if "neg_gt_base" not in WEEK_TOTAL_NUM[model_type][device_num]:
        WEEK_TOTAL_NUM[model_type][device_num]["neg_gt_base"] = 0
    if "neg_lt_base" not in WEEK_TOTAL_NUM[model_type][device_num]:
        WEEK_TOTAL_NUM[model_type][device_num]["neg_lt_base"] = 0
    if "fail_start" not in WEEK_TOTAL_NUM[model_type][device_num]:
        WEEK_TOTAL_NUM[model_type][device_num]["fail_start"] = 0
    if "fail_end" not in WEEK_TOTAL_NUM[model_type][device_num]:
        WEEK_TOTAL_NUM[model_type][device_num]["fail_end"] = 0


def compare_with_other_frame(job_res, total):
    """
     compare_with_other_frame
    """
    model_info = models.ModelInfo.objects.get(model_id = job_res.model_id)
    model_info_res = models.ModelInfo.objects.filter(~Q(frame='paddle'), 
                    model_name=model_info.model_name, model_repo=model_info.model_repo)
    #动态图、静态图：每个竞品框架只需提供一个竞品模型，竞品模型使用动态图、静态图训练都可以。
    #动转静：必须提供竞品动转静的模型。每个竞品框架是否都需要提供动转静竞品，详情找@张留杰 确认。
    model_ids = []
    for item in model_info_res:
        if model_info.model_type == "dynamicTostatic" and item.model_type == "dynamicTostatic":
            model_ids.append(item.model_id)
        if model_info.model_type != "dynamicTostatic" and item.model_type != "dynamicTostatic":
            model_ids.append(item.model_id)
    for frame_item in FRAMES:
        if frame_item not in total:
            total[frame_item] = {}
            total[frame_item]["gt"] = 0
            total[frame_item]["eq"] = 0
            total[frame_item]["lt"] = 0
            total[frame_item]["pf"] = 0
            total[frame_item]["of"] = 0
            total[frame_item]["no"] = 0

        job_result_res = models.JobResult.objects.filter(model_id__in=model_ids, batch_size=job_res.batch_size, 
                fp_mode=job_res.fp_mode, device_type=job_res.device_type, 
                device_num=job_res.device_num, run_mode=job_res.run_mode,
                cuda_version=job_res.cuda_version, cudnn_version=job_res.cudnn_version, 
                frame = frame_item).order_by("-frame_commit_dt")

        if (job_res.ips == 0 or job_res.ips == -1 or job_res.ips == -2 or 
                job_res.ips == -3 or job_res.ips == -4 or job_res.outlier == True):
            total[frame_item]["pf"] += 1
        elif len(job_result_res) == 0:
            total[frame_item]["no"] += 1
        else:
            speed_flag = False
            for res in job_result_res:
                if (res.ips == 0 or res.ips == -1 or res.ips == -2 or 
                     res.ips == -3 or res.ips == -4 or res.outlier == True):
                    continue
                else:
                    ratio = (job_res.ips - res.ips)/res.ips
                    if ratio > WEEK_WAVE_THRESHOLD:
                        total[frame_item]["gt"] += 1
                    elif ratio < -WEEK_WAVE_THRESHOLD:
                        total[frame_item]["lt"] += 1
                    else:
                        total[frame_item]["eq"] += 1
                    speed_flag = True
                    break
            if not speed_flag:
                total[frame_item]["of"] += 1


def compare_with_other_frame_total(start_job_res, job_res):
    """
      compare_with_other_frame_total
    """
    mi = models.ModelInfo.objects.get(model_id=start_job_res.model_id)
    if 'start' not in WEEK_TOTAL_NUM[mi.model_type][start_job_res.device_num]:
        WEEK_TOTAL_NUM[mi.model_type][start_job_res.device_num]['start'] = {}
    compare_with_other_frame(start_job_res, WEEK_TOTAL_NUM[mi.model_type][start_job_res.device_num]['start'])

    mi = models.ModelInfo.objects.get(model_id=job_res.model_id)
    if 'end' not in WEEK_TOTAL_NUM[mi.model_type][job_res.device_num]:
        WEEK_TOTAL_NUM[mi.model_type][job_res.device_num]['end'] = {}
    compare_with_other_frame(job_res, WEEK_TOTAL_NUM[mi.model_type][job_res.device_num]['end'])


def compare(start_job_res, job_res):
    """
    """
    mi = models.ModelInfo.objects.get(model_id=job_res.model_id)
    start_ips = start_job_res.ips
    start_base = start_job_res.speed_base_value
    end_ips = job_res.ips
    end_base = job_res.speed_base_value
    start_fail_flag = end_fail_flag = False
    res = {}
    if start_ips == 0 or start_ips == -1 or start_ips == -2 or start_ips == -3 or start_ips == -4 or start_ips == True:
         WEEK_TOTAL_NUM[mi.model_type][job_res.device_num]["fail_start"] += 1
         start_fail_flag = True
    if end_ips == 0 or end_ips == -1 or end_ips == -2 or end_ips == -3 or end_ips == -4 or end_ips == True:
         WEEK_TOTAL_NUM[mi.model_type][job_res.device_num]["fail_end"] += 1
         end_fail_flag = True
    res["val_range"] = "-"
    res["val_color"] = "white"
    res["base_range"] = "-"
    res["base_color"] = "white"
    try:
        res["base_range"] = round((end_base - start_base) / start_base, 4)
        res["base_color"] = "white"
        if res["base_range"] >= 0 and res["base_range"] <= WEEK_WAVE_THRESHOLD:
            WEEK_TOTAL_NUM[mi.model_type][job_res.device_num]["neg_lt_base"] += 1
        elif res["base_range"] > WEEK_WAVE_THRESHOLD:
            WEEK_TOTAL_NUM[mi.model_type][job_res.device_num]["neg_gt_base"] += 1
            res["base_color"] = "green"
        else:
            pass
    except Exception as e:
        print("except:", e)
    if start_fail_flag or end_fail_flag:
        return res
    try:
        res["val_range"] = round((end_ips - start_ips) / start_ips, 4)
        res["val_color"] = "white"
        if res["val_range"] < 0 and abs(res["val_range"]) > WEEK_WAVE_THRESHOLD:
            WEEK_TOTAL_NUM[mi.model_type][job_res.device_num]["pos_lt_st"] += 1
            res["val_color"] = 'red'
        elif res["val_range"] > WEEK_WAVE_THRESHOLD:
            WEEK_TOTAL_NUM[mi.model_type][job_res.device_num]["neg_gt_st"] += 1
            res["val_color"] = 'green'
        else:
            WEEK_TOTAL_NUM[mi.model_type][job_res.device_num]["between_st"] += 1
    except Exception as e:
        print("except:", e)
    return res


def get_compare_data(start_job_results, end_job_results):
    """
    start: start commit
    end:   end commit
    return compare data for weekly email
    """
    html_results = initial_data()
    for job_res in end_job_results:
        mi = models.ModelInfo.objects.get(model_id=job_res.model_id)
        construct_dict(mi.model_type, job_res.device_num)
        WEEK_TOTAL_NUM[mi.model_type][job_res.device_num]["total"] += 1
        start_job_res = start_job_results.filter(model_id=job_res.model_id,
                   batch_size=job_res.batch_size, fp_mode= job_res.fp_mode, run_mode = job_res.run_mode,
                   device_type=job_res.device_type, device_num=job_res.device_num, 
                   python_version= job_res.python_version,
                   cuda_version=job_res.cuda_version, cudnn_version=job_res.cudnn_version).order_by('-id')
        #同一个commit有多组数据情况，以最新的为准
        if len(start_job_res) == 0:
            continue
        res = compare(start_job_res.first(), job_res)
        #与竞品对比
        compare_with_other_frame_total(start_job_res.first(), job_res)
        bkcolor = "white"
        if job_res.device_num in BKCOLOR:
            bkcolor = BKCOLOR[job_res.device_num]
        if res["base_color"] == "white":
            res["base_color"] = bkcolor
        if res["val_color"] == "white":
            res["val_color"] = bkcolor
        cur_row_res= [dict(value=mi.model_name), dict(value=job_res.device_num, color=bkcolor), 
                         dict(value=job_res.ips, color=bkcolor),
                         dict(value=start_job_res.first().ips, color=bkcolor), 
                         dict(value="{:.2f}%".format(round(float(res["val_range"]) * 100, 2))
                                if type(res["val_range"])!= str else "-", color=res["val_color"]),
                         dict(value=job_res.speed_base_value, color=bkcolor), 
                         dict(value=start_job_res.first().speed_base_value, color=bkcolor), 
                         dict(value="{:.2f}%".format(round(float(res["base_range"]) * 100, 2))
                                if type(res["base_range"])!= str else "-", color=res["base_color"])]
        if mi.model_type == "static":
            html_results[WEEK_DICT_INDEX[4]]["data"].append(cur_row_res)
        elif mi.model_type == "dynamic":
            html_results[WEEK_DICT_INDEX[3]]["data"].append(cur_row_res)
        elif mi.model_type == "dynamicTostatic":
            html_results[WEEK_DICT_INDEX[5]]["data"].append(cur_row_res)
        else:
            pass
    #生成邮件总表
    total = pos_lt_st = neg_gt_st = between_st = neg_gt_base = neg_lt_base = fail_start = fail_end = 0
    for model_type, value in WEEK_TOTAL_NUM.items():
        des = ""
        if "static" == model_type:
            des = "静态图"
        elif "dynamic" == model_type:
            des = "动态图"
        elif "dynamicTostatic" == model_type:
            des = "动转静"
        else:
            des = "未知"
        for device_num, num in value.items():
            cur_row_res= [dict(value = des), dict(value = device_num),
                        dict(value= num["total"] if "total" in num else 0),
                        dict(value= num["pos_lt_st"] if "pos_lt_st" in num else 0, color='red'),
                        dict(value= num["neg_gt_st"] if "neg_gt_st" in num else 0),
                        dict(value= num["between_st"] if "between_st" in num else 0),
                        dict(value= num["neg_gt_base"] if "neg_gt_base" in num else 0),
                        dict(value= num["neg_lt_base"] if "neg_lt_base" in num else 0),
                        dict(value= num["fail_start"] if "fail_start" in num else 0),
                        dict(value= num["fail_end"] if "fail_end" in num else 0)]
            html_results[WEEK_DICT_INDEX[1]]["data"].append(cur_row_res)
            total += num["total"] if "total" in num else 0
            pos_lt_st += num["pos_lt_st"] if "pos_lt_st" in num else 0
            neg_gt_st += num["neg_gt_st"] if "neg_gt_st" in num else 0
            between_st += num["between_st"] if "between_st" in num else 0
            neg_gt_base += num["neg_gt_base"] if "neg_gt_base" in num else 0
            neg_lt_base += num["neg_lt_base"] if "neg_lt_base" in num else 0
            fail_start += num["fail_start"] if "fail_start" in num else 0
            fail_end += num["fail_end"] if "fail_end" in num else 0

            #生成竞品数据表
            total_dict = {}
            key_dict = {'start':"上周", 'end':"本周",}
            for index_item, value_item in key_dict.items():
                for frame_item in FRAMES:
                    gt = num[index_item][frame_item]["gt"]
                    eq = num[index_item][frame_item]["eq"]
                    lt = num[index_item][frame_item]["lt"]
                    pf = num[index_item][frame_item]["pf"]
                    of = num[index_item][frame_item]["of"]
                    no = num[index_item][frame_item]["no"]
                    if (gt + eq + lt) == 0:
                        rat = 0
                    else:
                        rat = (gt + eq) / (gt + eq + lt)
                    if frame_item not in total_dict:
                        total_dict[frame_item] = ""
                    total_dict[frame_item] += "{}:{}:{}:{}:{}:{}:{}({}%)".format(
                                        value_item, gt, eq, lt, pf, of, no, round(rat * 100))
            cur_row_res= [dict(value = des), dict(value = device_num), dict(value= total_dict["pytorch"]), 
                            dict(value= total_dict["tensorflow"]), dict(value= total_dict["oneflow"]), 
                            dict(value= total_dict["mxnet"]), dict(value= total_dict["mindspore"])]
            html_results[WEEK_DICT_INDEX[2]]["data"].append(cur_row_res)
 
    cur_row_res = [dict(value = "汇总"), dict(value = "-"), dict(value=total),
                dict(value=pos_lt_st), dict(value=neg_gt_st), dict(value=between_st), dict(value=neg_gt_base), 
                dict(value=neg_lt_base), dict(value=fail_start), 
                dict(value=fail_end)]
    html_results[WEEK_DICT_INDEX[1]]["data"].append(cur_row_res)
    return html_results


def get_compareres_bycommit(start_commit, end_commit, frame, device_type, cuda_version, cudnn_version, python_version):
    """
     get_compare_data_bycommit
    """
    start_job_results = models.JobResult.objects.filter(frame_commit=start_commit, frame=frame, 
                        device_type=device_type, 
                        cuda_version=cuda_version, cudnn_version=cudnn_version, python_version = python_version)
    end_job_results = models.JobResult.objects.filter(frame_commit=end_commit, frame=frame, device_type=device_type, 
                        cuda_version=cuda_version, cudnn_version=cudnn_version, python_version = python_version)
    return get_compare_data(start_job_results, end_job_results)
    

def get_compareres_bydate(start_date, end_date):
    """
     get_compareres_bydate
    """
    start_job_infos = models.JobInfo.objects.filter(task_date=start_date)
    start_job_ids = []
    for job_info in start_job_infos:
        if 'job-' in job_info.pdc_job_id:
            start_job_ids.append(job_info.pdc_job_id)
    start_job_results = models.JobResult.objects.filter(pdc_job_id__in=start_job_ids)
    end_job_infos = models.JobInfo.objects.filter(task_date=end_date)
    end_job_ids = []
    for job_info in end_job_infos:
        if 'job-' in job_info.pdc_job_id:
            end_job_ids.append(job_info.pdc_job_id)
    end_job_results = models.JobResult.objects.filter(pdc_job_id__in=end_job_ids)
    return get_compare_data(start_job_results, end_job_results)


def send_email_bycommit(start_commit, end_commit, frame, device_type, cuda_version, 
                                    cudnn_version, python_version, email_address):
    """
    send_email_bycommit
    """
    html_results = get_compareres_bycommit(start_commit, end_commit, frame, device_type, 
                    cuda_version, cudnn_version, python_version)
    env = dict(frame=frame, device_type=device_type, cuda_version=cuda_version, 
               cudnn_version=cudnn_version, python_version=python_version,
               start_commit=start_commit, end_commit=end_commit)
    email_t = EmailTemplate(env, html_results, os.getcwd())
    email_t.construct_weekly_email_content()
    #数据库中找到commit对应的日期
    start_commit_dt = models.JobResult.objects.filter(frame_commit=start_commit).first().frame_commit_dt
    end_commit_dt = models.JobResult.objects.filter(frame_commit=end_commit).first().frame_commit_dt
    subject = "【周级别邮件{}_{}_{}_CUDA{}】Benchmark每周信息统计".format(
            start_commit_dt.strftime("%Y-%m-%d"),
            end_commit_dt.strftime("%Y-%m-%d"), device_type, cuda_version)
    send_mail("paddle_benchmark@baidu.com", [email_address], subject, email_t.content)


def send_email_bydate(start_date, end_date):
    """
        send_email_bydate
    """
    html_results = get_compareres_bydate(start_date, end_date)
    env = dict(frame=frame, device_type=device_type, cuda_version=cuda_version, 
               cudnn_version=cudnn_version, python_version=python_version,
               start_commit=start_commit, end_commit=end_commit)
    email_t = EmailTemplate(env, html_results, os.getcwd())
    email_t.construct_weekly_email_content()
    subject = "【周级别邮件】运行结果报警，请查看"
    send_mail("paddle_benchmark@baidu.com", ["guolixin@baidu.com"], subject, email_t.content)


def send_week_email(frame, run_mode, device_type, cuda_version, cudnn_version, python_version, email_address):
    """
     send_week_email
    """
    job_res = models.JobResult.objects.filter(frame=frame, run_mode=run_mode, device_type=device_type,
                cuda_version= cuda_version, cudnn_version=cudnn_version, python_version=python_version)
    frame_commits = []
    items = job_res.order_by("-frame_commit_dt").values('frame_commit').distinct()
    for item in items:
        if item['frame_commit'] in ['', None]:
            continue
        if item['frame_commit'] not in frame_commits:
            frame_commits.append(item['frame_commit'])
    if len(frame_commits) < 5:
        print("frame commits counts < 5 ,can not get week email")
        return
    start_commit = frame_commits[4]
    end_commit = frame_commits[0]
    send_email_bycommit(start_commit, end_commit, frame, device_type, cuda_version, 
                        cudnn_version, python_version, email_address)
    

def week_email_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frame",
        type=str,
        default='paddle',
        help="frame")
    parser.add_argument(
        "--run_mode",
        type=str,
        default='DP',
        help="run_mode")
    parser.add_argument(
        "--device_type",
        type=str,
        default='V100',
        help="device_type")
    parser.add_argument(
        "--cuda_version",
        type=str,
        default='11.2',
        help="cuda_version")
    parser.add_argument(
        "--cudnn_version",
        type=str,
        default='8.1',
        help="frame")
    parser.add_argument(
        "--python_version",
        type=str,
        default='3.7',
        help="python_version")
    parser.add_argument(
        "--email_address",
        type=str,
        default='guolixin@baidu.com',
        help="email_address")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # start_commit = "63d4d05a39c5623c9b317ffffe94e8e0e2c16bc4"
    # end_commit = "80015c0667dbcfff2bfaaf28946abcb425244754"
    # frame = "paddle"
    # run_mode = "DP"
    # device_type = "V100"
    # cuda_version = "11.2"
    # cudnn_version = "8.1"
    # python_version = "3.7"
    # email_address = "guolixin@baidu.com"
    #send_week_email(frame, run_mode, device_type, cuda_version, cudnn_version, python_version, email_address)
    args = week_email_args()
    send_week_email(args.frame, args.run_mode, args.device_type, args.cuda_version, 
                        args.cudnn_version, args.python_version, args.email_address)
