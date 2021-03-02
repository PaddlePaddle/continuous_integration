
import os
import sys
import time
import subprocess
import logging

import cup
import py3nvml # only for python3.7

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def get_gpu_mem(gpu_id=0):
    """
    get gpu mem from gpu id
    Args:
        gpu_id (int): gpu id
    """
    py3nvml.py3nvml.nvmlInit()
    gpu_handle = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(gpu_id)
    gpu_mem_info = py3nvml.py3nvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    gpu_utilization_info = py3nvml.py3nvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    gpu_mem = {}
    gpu_mem['total(MB)'] = gpu_mem_info.total / 1024. ** 2
    gpu_mem['free(MB)'] = gpu_mem_info.free / 1024. ** 2
    gpu_mem['used(MB)'] = gpu_mem_info.used / 1024. ** 2
    gpu_mem['gpu_utilization_rate(%)'] = gpu_utilization_info.gpu
    gpu_mem['gpu_mem_utilization_rate(%)'] = gpu_utilization_info.memory
    py3nvml.py3nvml.nvmlShutdown()
    return gpu_mem

def get_pid(name):
    """
    get pid from process name
    """
    try:
        # pid = subprocess.check_output(["pidof", "-s", name])
        # pid = subprocess.getoutput(" ps -ef | grep '{0}' | grep -v 'grep' | awk '{printf $2}' ".format(name))
        pid = int(pid)
        # print(pid)
    except subprocess.CalledProcessError:
        logger.warning(subprocess.CalledProcessError)
        logger.error("No pid was detected, record process will end")
        pid = None
    return pid

def get_cpu_mem(pid):
    """
    get cpu mem from pid
    """
    if pid is None:
        logger.warning("process pid is None, end process")
        return None
    process = cup.res.Process(pid)
    mem_info = process.get_ext_memory_info()
    mem = {}
    mem['process_name'] = process.get_process_name()
    mem['rss(MB)'] = mem_info.rss / 1024. ** 2
    mem['vms(MB)'] = mem_info.vms / 1024. ** 2
    mem['shared(MB)'] = mem_info.shared / 1024. ** 2
    mem['dirty(MB)'] = mem_info.dirty / 1024. ** 2
    mem['cpu_usage(%)'] = cup.res.linux.get_cpu_usage(intvl_in_sec=1).usr
    return mem


def summary(cpu_mem_lists, gpu_mem_lists=None):
    """
    return reports of cpu and gpu info
    """
    cpu_reports = {}
    gpu_reports = {}

    cpu_reports['process_name'] = cpu_mem_lists[0]['process_name']
    cpu_reports['rss(MB)'] = max([float(i['rss(MB)']) for i in cpu_mem_lists])
    cpu_reports['vms(MB)'] = max([float(i['vms(MB)']) for i in cpu_mem_lists])
    cpu_reports['shared(MB)'] = max([float(i['shared(MB)']) for i in cpu_mem_lists])
    cpu_reports['dirty(MB)'] = max([float(i['dirty(MB)']) for i in cpu_mem_lists])
    cpu_reports['cpu_usage(%)'] = max([float(i['cpu_usage(%)']) for i in cpu_mem_lists])

    logger.info("----------------------- Res info -----------------------")
    logger.info("process_name: {0}, cpu rss(MB): {1}, \
vms(MB): {2}, shared(MB): {3}, dirty(MB): {4}, \
cpu_usage(%): {5} ".format(cpu_reports['process_name'],
                           cpu_reports['rss(MB)'],
                           cpu_reports['vms(MB)'],
                           cpu_reports['shared(MB)'],
                           cpu_reports['dirty(MB)'],
                           cpu_reports['cpu_usage(%)']))

    if gpu_mem_lists:
        logger.info("=== gpu info was recorded ===")
        gpu_reports['gpu_id'] = int(os.environ.get("CUDA_VISIBLE_DEVICES"))
        gpu_reports['total(MB)'] = max([float(i['total(MB)']) for i in gpu_mem_lists])
        gpu_reports['free(MB)'] = max([float(i['free(MB)']) for i in gpu_mem_lists])
        gpu_reports['used(MB)'] = max([float(i['used(MB)']) for i in gpu_mem_lists])
        gpu_reports['gpu_utilization_rate(%)'] = max([float(i['gpu_utilization_rate(%)']) for i in gpu_mem_lists])
        gpu_reports['gpu_mem_utilization_rate(%)'] = max([float(i['gpu_mem_utilization_rate(%)']) for i in gpu_mem_lists])

        logger.info("gpu_id: {0}, total(MB): {1}, \
free(MB): {2}, used(MB): {3}, gpu_utilization_rate(%): {4}, \
gpu_mem_utilization_rate(%): {5} ".format(gpu_reports['gpu_id'],
                                          gpu_reports['total(MB)'],
                                          gpu_reports['free(MB)'],
                                          gpu_reports['used(MB)'],
                                          gpu_reports['gpu_utilization_rate(%)'],
                                          gpu_reports['gpu_mem_utilization_rate(%)']))
        return cpu_reports, gpu_reports
    else:
        return cpu_reports, None

def record_from_pipe():
    """
    record_from_pipe
    """
    process = sys.argv[1]
    time.sleep(0.5)

    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES") # 0,1,2,3
    gpu_id_lists = gpu_ids.split(',')
    if len(gpu_id_lists) > 1:
        logger.warning("more than one CUDA_VISIBLE_DEVICES was set")
    else:
        gpu_id = int(gpu_ids)

    cpu_mem_lists = []
    gpu_mem_lists = []
    while get_pid(process):
        pid = get_pid(process)
        if pid:
            cpu_mem = get_cpu_mem(pid)
            gpu_mem = get_gpu_mem(gpu_id)
            cpu_mem_lists.append(cpu_mem)
            gpu_mem_lists.append(gpu_mem)
        else:
            logger.warning("==== process pid is None, end recording ===")
            break
        time.sleep(0.5)

    cpu_reports, gpu_reports = summary(cpu_mem_lists, gpu_mem_lists)
    # print(cpu_reports)
    # print(gpu_reports)

def record_from_pid(pid : int):
    """
    record_from_pid
    """
    time.sleep(0.5)

    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES") # 0,1,2,3
    gpu_id_lists = gpu_ids.split(',')
    if len(gpu_id_lists) > 1:
        logger.warning("more than one CUDA_VISIBLE_DEVICES was set")
    else:
        gpu_id = int(gpu_ids)
    cpu_mem_lists = []
    gpu_mem_lists = []
    while True:
        cpu_mem = get_cpu_mem(pid)
        gpu_mem = get_gpu_mem(gpu_id)
        cpu_mem_lists.append(cpu_mem)
        gpu_mem_lists.append(gpu_mem)
        cpu_reports, gpu_reports = summary(cpu_mem_lists, gpu_mem_lists)
        time.sleep(0.5)

if __name__ == "__main__":
    record_from_pipe()