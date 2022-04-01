# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import time
import subprocess
import logging

import cup
import py3nvml  # only for python3.7

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def get_gpu_mem(gpu_id=0) -> dict:
    """
    get gpu mem from gpu id
    Args:
        gpu_id (int): gpu id
    Returns:
        gpu_mem (dict): gpu memory of target gpu
    """
    py3nvml.py3nvml.nvmlInit()
    gpu_handle = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(gpu_id)
    gpu_mem_info = py3nvml.py3nvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    gpu_mem = {}
    gpu_mem['total(MB)'] = gpu_mem_info.total / 1024.**2
    gpu_mem['free(MB)'] = gpu_mem_info.free / 1024.**2
    gpu_mem['used(MB)'] = gpu_mem_info.used / 1024.**2
    py3nvml.py3nvml.nvmlShutdown()
    return gpu_mem


def get_pid(name: str) -> int:
    """
    get pid from process name
    Args:
        name (str): process name
    Returns:
        pid (int): process id
    """
    try:
        pid = subprocess.check_output(["pidof", "-s", name])
        pid = int(pid)
    except subprocess.CalledProcessError:
        logger.warning(subprocess.CalledProcessError)
        pid = None
    return pid


def get_cpu_mem(pid: int) -> dict:
    """
    get cpu mem from pid(use get_pid to retrive pid num first)
    Args:
        pid (int): process id
    Returns:
        mem (dict): [description]
    """
    if pid is None:
        logger.warning("process pid is None, end process")
        mem = None
        return mem

    process = cup.res.Process(pid)
    mem_info = process.get_ext_memory_info()
    mem = {}
    mem['rss(MB)'] = mem_info.rss / 1024.**2
    mem['vms(MB)'] = mem_info.vms / 1024.**2
    mem['shared(MB)'] = mem_info.shared / 1024.**2
    mem['dirty(MB)'] = mem_info.dirty / 1024.**2
    return mem
