#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
#======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
#======================================================================

"""
@Desc: dist_base_fleet module
@File: dist_base_fleet.py
@Author: liangjinhua
@Date: 2019/8/26 19:21
"""

from __future__ import print_function

import paddle
import math
import time
import numpy as np
import paddle.fluid as fluid
import os
import sys
sys.path.append('./ctr')
import py_reader_generator as py_reader1
# from cts_test.dist_fleet.reader_generator import ctr_py_reader_generator as py_reader1
from dist_base_fleet import runtime_main
from dist_base_fleet import FleetDistRunnerBase

params = {
    "is_first_trainer":True,
    "model_path":"dist_model_ctr",
    "is_pyreader_train":True,
    "is_dataset_train":False
}

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1
np.random.seed(1)


class TestDistCTR(FleetDistRunnerBase):
    """distCTR"""
    def input_data(self):
        """def input"""
        dense_feature_dim = 13
        self.dense_input = fluid.layers.data(
            name="dense_input", shape=[dense_feature_dim], dtype='float32')

        self.sparse_input_ids = [
            fluid.layers.data(
                name="C" + str(i), shape=[1], lod_level=1, dtype='int64')
            for i in range(1, 27)
        ]

        self.label = fluid.layers.data(
            name='label', shape=[1], dtype='int64')

        self._words = [self.dense_input
                       ] + self.sparse_input_ids + [self.label]
        return self._words

    def py_reader(self):
        """py reader"""
        py_reader = fluid.layers.create_py_reader_by_data(
            capacity=64,
            feed_list=self._words,
            name='py_reader',
            use_double_buffer=False)
        return py_reader

    def dataset_reader(self):
        """dataset reader"""
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var([self.dense_input] + self.sparse_input_ids +
                            [self.label])
        pipe_command = "python ./ctr/dataset_generator.py"
        dataset.set_pipe_command(pipe_command)
        dataset.set_batch_size(4)
        thread_num = int(2)
        dataset.set_thread(thread_num)
        return dataset

    def net(self, args=None):
        """net """  
        self.inputs = self.input_data()         
        if not args.run_params.get("run_from_dataset", False):
            self.pyreader = self.py_reader()
            self.inputs = fluid.layers.read_file(self.pyreader)

        sparse_feature_dim = 1000001
        embedding_size = 10
        words = self.inputs
        def embedding_layer(input):
            return fluid.layers.embedding(
                input=input,
                is_sparse=True,
                is_distributed=False,
                size=[sparse_feature_dim, embedding_size],
                param_attr=fluid.ParamAttr(
                    name="SparseFeatFactors",
                    initializer=fluid.initializer.Uniform()))

        sparse_embed_seq = list(map(embedding_layer, words[1:-1]))
        concated = fluid.layers.concat(
            sparse_embed_seq + words[0:1], axis=1)

        fc1 = fluid.layers.fc(input=concated, size=400, act='relu',
                              param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                  scale=1 / math.sqrt(concated.shape[1]))))
        fc2 = fluid.layers.fc(input=fc1, size=400, act='relu',
                              param_attr=fluid.ParamAttr(
                                  initializer=fluid.initializer.Normal(
                                      scale=1 / math.sqrt(fc1.shape[1]))))
        fc3 = fluid.layers.fc(input=fc2, size=400, act='relu',
                              param_attr=fluid.ParamAttr(
                                  initializer=fluid.initializer.Normal(
                                      scale=1 / math.sqrt(fc2.shape[1]))))
        predict = fluid.layers.fc(input=fc3, size=2, act='softmax',
                                  param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                                      scale=1 / math.sqrt(fc3.shape[1]))))

        cost = fluid.layers.cross_entropy(input=predict, label=words[-1])
        self.avg_cost = fluid.layers.reduce_sum(cost)
        accuracy = fluid.layers.accuracy(input=predict, label=words[-1])
        auc_var, batch_auc_var, auc_states = \
            fluid.layers.auc(input=predict, label=words[-1], num_thresholds=2 ** 12, slide_steps=20)
        return self.avg_cost

    def check_model_right(self, dirname):
        """check """
        model_filename = os.path.join(dirname, "__model__")

        with open(model_filename, "rb") as f:
            program_desc_str = f.read()

        program = fluid.Program.parse_from_string(program_desc_str)
        with open(os.path.join(dirname, "__model__.proto"), "w") as wn:
            wn.write(str(program))

    def do_training(self, fleet, args):
        """training"""
        exe = fluid.Executor(fluid.CPUPlace())
        fleet.init_worker()
        exe.run(fleet.startup_program)
        train_generator = py_reader1.CriteoDataset(1000001)
        file_list = [str('data/dist_data/ctr_data/part-100')] * 2
        train_reader = paddle.batch(
                train_generator.train(file_list,
                                      args.trainers,
                                      args.current_id),
                                       batch_size=4)
        self.pyreader.decorate_paddle_reader(train_reader)       
        if os.getenv("PADDLE_COMPATIBILITY_CHECK", False):
            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_threads = int(2)
            build_strategy = fluid.BuildStrategy()
            build_strategy.async_mode = self.async_mode
            if args.run_params["sync_mode"] == "async":
                build_strategy.memory_optimize = False
            if args.run_params['cpu_num'] > 1:
                build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        else:
            build_strategy = self.strategy.get_build_strategy()
            if args.run_params["sync_mode"] == "async":
                build_strategy.memory_optimize = False
                self.strategy.set_build_strategy(build_strategy)   
            exec_strategy = self.strategy.get_execute_strategy()
     
        compiled_prog = fluid.compiler.CompiledProgram(
            fleet.main_program).with_data_parallel(
            loss_name=self.avg_cost.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

        # Notice: py_reader should use try & catch EOFException method to enter the dataset
        # reader.start() must declare in advance
        self.pyreader.start()
        train_info = []
        batch_id = 0
        try:
            while True:
                avg_cost = exe.run(program=compiled_prog, fetch_list=[self.avg_cost.name])
                avg_cost = np.mean(avg_cost)
                train_info.append(avg_cost)
                batch_id += 1
                if params["is_first_trainer"]:
                    if params["is_pyreader_train"]:
                        model_path = str(params["model_path"] + "/final" + "_pyreader")
                        fleet.save_persistables(
                            executor=fluid.Executor(fluid.CPUPlace()),
                            dirname=model_path)
                    elif params["is_dataset_train"]:
                        model_path = str(params["model_path"] + '/final' + "_dataset")
                        fleet.save_persistables(
                            executor=fluid.Executor(fluid.CPUPlace()),
                            dirname=model_path)
                    else:
                        raise ValueError("Program must has Date feed method: is_pyreader_train / is_dataset_train")
                if batch_id == 5:
                    break
        except fluid.core.EOFException:
            self.pyreader.reset()
        fleet.stop_worker()
        return train_info

    def do_training_from_dataset(self, fleet, args):
        """ training_from_dataset """
        exe = fluid.Executor(fluid.CPUPlace())
        fleet.init_worker()
        exe.run(fleet.startup_program)
        dataset = self.dataset_reader()
        file_list = [str('data/dist_data/ctr_data/part-100')] * 2
        for epoch in range(1):
            dataset.set_filelist(file_list)
            var_dict = {"loss": self.avg_cost}
            train_info = []
            class FetchVars(fluid.executor.FetchHandler):
                def __init__(self, var_dict=None, period_secs=5):
                    super(FetchVars, self).__init__(var_dict, period_secs=5)
                def handler(self, res_dict):
                    if len(train_info) < 6:
                        train_info.extend(res_dict['loss'].tolist())
                    
            exe.train_from_dataset(program=fleet.main_program,
                                   dataset=dataset,
                                   fetch_handler=FetchVars(var_dict))
            
            return train_info


if __name__ == "__main__":
    runtime_main(TestDistCTR)