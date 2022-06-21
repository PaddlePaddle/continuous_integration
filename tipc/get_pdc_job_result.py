import os
import sys
import json
import time
import requests

JOBS_INFO = {}

def get_pdc_job_id(model_job_file):
    """
    """
    with open(model_job_file, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            model, job_id = line.strip().split(",")
            if "job-" not in job_id:
                continue
            JOBS_INFO[model] = {"job_id": job_id,
                                "used_time": "",
                                "status": "UNK",
                                "log": 0}


def get_pdc_log(job_id):
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
    #此处需要按照jobName找到相应的log
    result_addr = ("http://" + ip + "/filetree?action=cat&path=/home/work/containers_backup/" +
               code + "/root/paddlejob/workspace/log/train.log")
    try:
        result_data = requests.get(result_addr).content
    except Exception as e:
        print("get_log_thread:request get content except:", e)
        return False
    if result_data != bytes():
        with open("pdc_" + job_id + "_" + job_name + "_train.log", "wb") as fout:
            fout.write(result_data)
    return True


def get_log():
    """
    """
    for model, infos in JOBS_INFO.items():
        print("get_log:", model, infos)
        job_id = infos["job_id"]
        get_pdc_log(job_id) 


def get_job_status(pdc_job_id):
    '''
      由job_id获取pdc运行状态,以及usedTime
    '''
    if "job-" not in pdc_job_id:
        print("watch_job_thread:job id error")
        return "error", ""
    while True:
       output = os.popen("paddlecloud job state %s" % (pdc_job_id)).read()
       try:
           output_dict = json.loads(output)
           break
       except Exception as e:
           print("watch_job_thread:paddlecloud job state except:", e)
           time.sleep(300)
           continue
    status = output_dict["jobStatus"]
    #print(status)
    used_time = ""
    if "usedTime" in output_dict.keys():
         used_time = output_dict["usedTime"]
    return status, used_time


def update_job():
    for model, infos in JOBS_INFO.items():
        status, used_time = get_job_status(infos["job_id"])
        JOBS_INFO[model]["used_time"] = used_time
        if status.find("success") != -1:
            JOBS_INFO[model]["status"] = "success"
        elif status.find("fail") != -1:
            JOBS_INFO[model]["status"] = "fail"
        elif status.find("killed") != -1:
            JOBS_INFO[model]["status"] = "killed"
        elif status.find("killing") != -1:
            JOBS_INFO[model]["status"] = "killing"
        elif status.find("schedule") != -1:
            JOBS_INFO[model]["status"] = "schedule"
        elif status.find("submit") != -1:
            JOBS_INFO[model]["status"] = "submit"
        elif status.find("queue") != -1:
            JOBS_INFO[model]["status"] = "queue"
        elif status.find("running") != -1:
            JOBS_INFO[model]["status"] = "running"
            usetime_list = used_time.split(" hour")
            if len(usetime_list) > 1:
                hour = usetime_list[0]
                if int(hour) >= 2:
                    os.popen("paddlecloud job kill %s" % (job_id))
                    print("watch_job_thread:kill jobid is %s" % (job_id))
        else:
            pass


def watch_job_thread_finished():
    '''
    '''
    finished = True
    for model, infos in JOBS_INFO.items():
        print("watch_job:", model, infos)
        if infos["status"] in ["submit", "queue", "running", "schedule", "UNK"]:
             finished = False
             break
    return finished


def watch_job():
    '''
      如果这次任务没有结束，就一直监控此任务
    '''
    while True:
        if not watch_job_thread_finished():
            print("watch job thread running...")
            update_job()
            time.sleep(300)
        else:
            break


if __name__ == "__main__":
    model_job_file = sys.argv[1]
    get_pdc_job_id(model_job_file)
    watch_job() 
    get_log()
    
