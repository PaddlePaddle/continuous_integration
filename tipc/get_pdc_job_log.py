import os
import sys
import json
import time
import requests
from pathlib import Path


def get_job_ip(job_id):
    """
    """
    if "job-" not in job_id:
        print("get_log_thread:job id error")
        return False
    output = os.popen("paddlecloud job info %s" % (job_id)).read()
    try:
        output_dict = json.loads(output)
    except Exception as e:
        print("get_log_thread:paddlecloud job info except:", e)
        return False
    job_name = output_dict["jobName"]
    trainerList = output_dict["trainerList"]
    trainer_index = 0
    trainer_name = "trainer-0"
    trainer = None
    for t in trainerList:
        if t["resourceName"].find(trainer_name) != -1:
            trainer = t
    if trainer == None:
        print("get_log_thread: pdc job trainer error")
        return False
    addr = trainer["logUrl"]
    addr_list = addr.split("/")
    ip = addr_list[2]
    code = addr_list[5]
    return ip, code 


def get_pdc_log(ip, code, repo, log_path):
    """
    """
    print(log_path)
    tag = True
    #此处需要按照jobName找到相应的log
    result_addr = ("http://" + ip + "/filetree?action=cat&path=/home/work/containers_backup/" +
               code + "/root/paddlejob/workspace/env_run/Paddle/" + repo + "/" + log_path)
    try:
        result_data = requests.get(result_addr).content
        print(result_data)
    except Exception as e:
        print("get_log_thread:request get content except:", e)
        tag = False
    if result_data != bytes():
        with open(log_path, "wb") as fout:
            fout.write(result_data)
    return tag


def get_log(repo):
    """
    """
    # 遍历 *train.log
    for item in Path(os.getcwd()).glob("pdc_job-*_train.log"):
        print(str(item))
        job_id = str(item).split("/")[-1].split("_")[1] 
        print(job_id)
        ip, code = get_job_ip(job_id)
        # 遍历 RESULT, 获取log_path
        with open(str(item), "r") as fin:
            lines = fin.readlines()
            for line in lines:
                
                if ("Run successfully with command" in line) or ("Run failed with command" in line): 
                    log_path = line.split(' - ')[3].strip().strip("./")
                    get_pdc_log(ip, code, repo, log_path)
        

if __name__ == "__main__":
    REPO = sys.argv[1]
    get_log(REPO)
    
