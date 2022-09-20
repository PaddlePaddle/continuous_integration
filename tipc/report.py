#coding=utf-8

import sys
import smtplib
from email.mime.text import MIMEText
from email.header    import Header

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
    "content": "",
    "sender_addr": "",
    "receiver_addr": "",
    "subject": "",
}


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


def get_info():
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
            if "successfully" in tmp[0]:
                tag = "success"
                res["success_cases_num"] += 1
            else:
                tag = "failed"
                res["failed_cases_num"] += 1
            model_name = tmp[1].strip()
            case = tmp[2]
            stage = ""
            if ("train.py --test-only" in case) or ("main.py --test" in case):
                stage = "eval"
            elif ("train.py" in case) or ("main.py --validat" in case) or ("tools/main.py" in case):
                stage = "train"
            elif ("export_model.py" in case) or ("export.py" in case) or ("to_static.py" in case):
                stage = "dygraph2static"
            elif ("infer.py" in case) or ("predict_det.py" in case):
                stage = "inference"
            else:
                stage = "inference"
            if model_name not in res["models_status"].keys():
                res["models_status"].setdefault(model_name, [])
            res["models_status"][model_name].append({"status": tag, "case": case, "stage": stage})
            if model_name not in res["model_func"].keys():
                res["model_func"].setdefault(model_name, {"train": {"success": 0, "failed": 0}, "eval": {"success": 0, "failed": 0}, "dygraph2static": {"success": 0, "failed": 0}, "inference": {"success": 0, "failed": 0}, "UNK": {"success": 0, "failed": 0}})
            res["model_func"][model_name][stage][tag] += 1
    for model, infos in res["models_status"].items():
        if model in res["timeout_models"]:
            continue
        tag = "success"
        for item in infos:
            if item["status"] == "failed":
                tag = "failed"
                break
        if tag == "failed":
            res["failed_num"] += 1
            res["failed_models"].append(model)
        else:
            res["success_num"] += 1
            res["success_models"].append(model)
    #res["success_num"] = res["total_num"] - res["timeout_num"] - res["failed_num"]
    res["success_models"].sort()
    res["failed_models"].sort()
    res["timeout_models"].sort()


def print_result():
    """
    """
    msg = "=" * 50
    msg += "\n"
    msg += "TOTAL: {} models\n\n".format(str(res["total_num"]))
    msg += "SUCCESS: {} models\n\n".format(str(res["success_num"]))
    msg += " ".join(res["success_models"])
    msg += "\n\n"
    msg += "TIMEOUT: {} models:\n".format(str(res["timeout_num"]))
    msg += " ".join(res["timeout_models"])
    msg += "\n\n"
    msg += "FAILED: {} models:\n".format(str(res["failed_num"]))
    msg += " ".join(res["failed_models"])
    if res["failed_cases_num"] > 0:
        msg += "\n{} cases failed:\n".format(str(res["failed_cases_num"]))
        #for model in res["failed_models"]:
        #    for item in res["models_status"][model]:
        #        if item["status"] == "failed":
        #            msg += "Failed: {} {} {}\n".format(model, item["stage"], item["case"])
    msg += "=" * 50
    print(msg)


def send_mail(sender_addr, receiver_addr, repo, chain, proxy):
    content = """
<html>
    <body>
        <div style="text-align:center;">
        </div>
"""
    # table1
    content += """
        <table border="1" align=center>
        <caption bgcolor="#989898">任务执行情况汇总</caption>
        <tr><td></td><td>成功</td><td>失败</td><td>超时</td></tr>
"""
    content += """
        <tr><td>模型</td><td>{}</td><td>{}</td><td>{}</td></tr>
""".format(res["success_num"], res["failed_num"], res["timeout_num"])
    content += """
        <tr><td>case</td><td>{}</td><td>{}</td><td>-</td></tr>
""".format(res["success_cases_num"], res["failed_cases_num"])
    content += """
        </table>
        <br><br>
"""

    # table2
    content += """
        <table border="1" align=center>
        <caption bgcolor="#989898">分功能汇总</caption>
        <tr><td></td><td>训练</td><td>评估</td><td>动转静</td><td>推理</td></tr>
"""
    for model, infos in res["model_func"].items():
        content += """
            <tr><td>{}</td><td>成功:{} 失败:{}</td><td>成功:{} 失败:{}</td><td>成功:{} 失败:{}</td><td>成功:{} 失败:{}</td></tr>
    """.format(model, infos["train"]["success"], infos["train"]["failed"], infos["eval"]["success"], infos["eval"]["failed"], infos["dygraph2static"]["success"], infos["dygraph2static"]["failed"], infos["inference"]["success"], infos["inference"]["failed"])
    content += """
        </table>
        <br><br>
"""
    # table3
    content += """
        <table border="1" align=center>
        <caption bgcolor="#989898">失败列表</caption>
        <tr><td>模型</td><td>case</td></tr>
"""
    for model in res["failed_models"]:
        for item in res["models_status"][model]:
            if item["status"] == "failed":
                content += """
        <tr><td>{}</td><td>{}</td></tr>
""".format(model, item["case"])
    content += """
        </table>

    </body>
</html>
"""
    res["content"] = content
    res["sender_addr"] = sender_addr
    res["receiver_addr"] = receiver_addr
    res["subject"] = "【TIPC:{}:{}】执行结果".format(repo, chain)
    mail(res["sender_addr"], res["receiver_addr"], res["subject"], res["content"], proxy)


if __name__ == "__main__":
    repo = sys.argv[1]
    chain = sys.argv[2]
    sender_addr = sys.argv[3]
    receiver_addr = sys.argv[4]
    proxy = sys.argv[5]
    get_info()
    print_result()
    send_mail(sender_addr, receiver_addr, repo, chain, proxy)
