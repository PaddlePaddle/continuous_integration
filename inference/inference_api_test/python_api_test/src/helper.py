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
import os
import json
import logging
import numpy as np

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level = logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

class Record(object):
    """
    one record for one sample
    """
    def __init__(self):
        """
        __init__
        """
        data = None
        shape = None
        lod = None

    def process_file(self,
                     file_name,
                     sample_size,
                     sep=' '):
        """
        process txt file by using generator
        Args:
            file_name(strings):
            sample_size(int):
            sep(strings): space, comma, etc.
        """
        with open(file_name) as file_handler:
            batch_data = []
            for i, block in enumerate(file_handler.readlines()):
                if i == file_handler.tell():
                    logger.info("file handler reach end of the file, length is [%s]" % i)
                    break
                data = np.array(block.strip().split(sep), dtype="float32")
                batch_data.append(data)
                if len(batch_data) == sample_size:
                    yield batch_data
                    batch_data = []

    def load_data_from_json(self, json_info, sample_size=1):
        """
        load one record
        Args:
            json_info(class|JsonInfo): a class whith json information
            sample_size(int): if data.txt has multiple lines, use sample_size to control
        Returns:
            record(class|Record): a class describes all info needed by Tensor
        """
        record = Record()
        # init data
        file_name = json_info.data
        for id, value in enumerate(self.process_file(file_name, sample_size)):
            data = np.array(value).reshape(json_info.shape).astype(json_info.dtype)
            if id == 1:
                break
        record.data = data
        # init shape
        record.shape = json_info.shape
        if hasattr(json_info, 'lod'):
            # init lod
            record.lod = json_info.lod
        return record


class JsonInfo(object):
    """
    Store whole Json Information
    """
    def __init__(self):
        """
        __init__
        """
        data = None
        shape = None
        dtype = None
        lod = None

    def parse_json(self, json_dir):
        """
        parse json information
        Args:
            json_dir(string|*.json): a *.json file
        Returns:
            inputs_info(list|[JsonInfo, JsonInfo]): a list contain several JsonInfo
        """
        if not os.path.exists(json_dir):
            raise Exception('data json file path [%s] invalid! file do not exist' % json_dir)
        inputs_info = []
        with open(json_dir, 'r') as f:
            sample = json.load(f)
        
        # split json path
        json_path = os.path.split(json_dir)[0]
        
        for key in sample.keys():
            json_info = JsonInfo()
            # path of data0.txt should be the same with data.json
            data_path = os.path.split(sample[key]['data'])[-1]
            json_info.data = os.path.join(json_path, data_path)
            json_info.shape = sample[key]['shape']
            json_info.dtype = sample[key]['dtype']
            if ('lod' in sample[key]):
                json_info.lod = sample[key]['lod']
            inputs_info.append(json_info)
        return inputs_info