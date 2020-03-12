#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019-12-25 19:49
# @Author  : liyang109

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import math
import time
import numpy as np
import paddle.fluid as fluid
import os
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
import logging
import mpi4py
from mpi4py import rc

rc.finalize = False
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.WARNING)
# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1
feature_num = 49431


class TestDistPslib():
    """test pslib"""
    def net(self, args=None):
        """net"""
        label = fluid.layers.data(name="click", shape=[-1, 1], dtype="int64", lod_level=1, append_batch_size=False)
        data = fluid.layers.data(name="1", shape=[1], dtype="int64", lod_level=1)

        # logistic regression
        emb = fluid.layers.embedding(
            input=data,
            size=[feature_num, 2],
            is_sparse=True,
            is_distributed=True,
            param_attr=fluid.ParamAttr(name="embedding"))
        emb1 = fluid.layers.slice(emb, axes=[1], starts=[0], ends=[1])

        bow = fluid.layers.sequence_pool(input=emb1, pool_type='sum')
        prediction = fluid.layers.sigmoid(bow)
        # cost auc_var
        cost = fluid.layers.log_loss(input=prediction, label=fluid.layers.cast(x=label, dtype='float32'))
        self.avg_cost = fluid.layers.mean(x=cost)
        auc_var, batch_auc_var, auc_states = fluid.layers.auc(
            input=prediction, label=label, num_thresholds=2 ** 12, slide_steps=20)
        return self.avg_cost

    def do_training(self, args=None):
        """do training"""
        avg_cost = self.net()
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        fleet.init(exe)
        # optimizer
        optimizer = fluid.optimizer.Adam(learning_rate=0.001)
        # 加入 fleet distributed_optimizer 加入分布式策略配置及多机优化
        optimizer = fleet.distributed_optimizer(optimizer, strategy={"fleet_desc_file": "./pslib/fleet_desc.prototxt"})
        optimizer.minimize(avg_cost)
        train_info = []
        # 启动server
        if fleet.is_server():
            fleet.init_server()
            fleet.run_server()
        # 启动worker
        if fleet.is_worker():
            train_data_path = './data/pslib/train_data'
            train_data_files = []
            for filename in os.listdir(train_data_path):
                train_data_files.append(os.path.join(train_data_path, filename))
            # fleet dataset
            label = fluid.layers.data(name="click", shape=[-1, 1], dtype="int64", lod_level=1, append_batch_size=False)
            data = fluid.layers.data(name="1", shape=[1], dtype="int64", lod_level=1)
            dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
            dataset.set_use_var([label, data])
            dataset.set_pipe_command("./python/bin/python ./pslib/dataset_generator.py")
            dataset.set_batch_size(32)
            dataset.set_thread(3)
            dataset.set_filelist(train_data_files)
            # 把数据读到内存
            dataset.load_into_memory()
            # 本地shuffle
            dataset.local_shuffle()
            # 初始化worker配置
            fleet.init_worker()
            exe.run(fluid.default_startup_program())
            PASS_NUM = 1
            for pass_id in range(PASS_NUM):
                var_dict = {"loss": avg_cost}
                global var_dict

                class FetchVars(fluid.executor.FetchHandler):
                    def __init__(self, var_dict=None, period_secs=2):
                        super(FetchVars, self).__init__(var_dict, period_secs=2)

                    def handler(self, res_dict):
                        train_info.extend(res_dict["loss"])
                        print(train_info)

                exe.train_from_dataset(
                    program=fluid.default_main_program(),
                    dataset=dataset,
                    fetch_handler=FetchVars(var_dict))
                dataset.release_memory()
        fleet.stop_worker()
        return train_info


if __name__ == '__main__':
    c = TestDistPslib()
    c.do_training()