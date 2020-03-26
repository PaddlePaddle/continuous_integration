#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
#======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
#======================================================================
"""
@Desc: dist_fleet_resnet module
@File: dist_fleet_resnet.py
@Author: liangjinhua
@Date: 2019/8/26 19:21
"""
from __future__ import print_function

import paddle.fluid as fluid
import os
import sys
import paddle
import json
import numpy as np
from dist_base_fleet import runtime_main
from dist_base_fleet import FleetDistRunnerBase
import thirdparty.image_classfication.utils.reader_cv2 as reader
sys.path.append("./thirdparty/image_classfication")

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1
np.random.seed(1)

DATA_DIR = './thirdparty/data/dist_data/ImageNet'


class TestDistResNet50(FleetDistRunnerBase):
    """Test ResNet50 fleet."""

    def __init__(self):
        FleetDistRunnerBase.__init__(self, batch_num=5, batch_size=1000)
        self.model_name = 'ResNet50'
        self.batch_size = 8
        self.batch_num = 5

    def net(self, args=None):
        """
        resnet struct.
        Args:
            fleet:
            args (ArgumentParser): run args to config dist fleet.
        Returns:
            tuple: the return value contains avg_cost, py_reader
        """
        from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
        from thirdparty.image_classfication.models.resnet import ResNet50
        from thirdparty.image_classfication.train import parser
        from thirdparty.image_classfication.train import optimizer_setting
        parser.add_argument(
            '--update_method',
            type=str,
            required=True,
            choices=['pserver', 'nccl'])
        parser.add_argument(
            '--role', type=str, required=True, choices=['pserver', 'trainer'])
        parser.add_argument(
            '--endpoints', type=str, required=False, default="")
        parser.add_argument(
            '--current_id', type=int, required=False, default=0)
        parser.add_argument('--trainers', type=int, required=False, default=1)
        # parser.add_argument('--sync_mode', action='store_true')
        parser.add_argument(
            '--run_params', type=str, required=False, default='{}')
        args = parser.parse_args()
        args.run_params = json.loads(args.run_params)
        image_shape = [3, 224, 224]
        scale_loss = 1.0
        self.py_reader = fluid.layers.py_reader(
            capacity=16,
            shapes=[[-1] + image_shape, [-1, 1]],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            use_double_buffer=True)
        image, label = fluid.layers.read_file(self.py_reader)
        run_model = ResNet50()
        out = run_model.net(image, 4)
        softmax_out = fluid.layers.softmax(out, use_cudnn=False)
        cost, prob = fluid.layers.softmax_with_cross_entropy(
            out, label, return_softmax=True)
        self.avg_cost = fluid.layers.mean(cost)

        params = run_model.params
        params["total_images"] = args.total_images
        params["lr"] = 1e-5
        params["num_epochs"] = args.num_epochs
        params["learning_strategy"]["batch_size"] = args.batch_size
        params["learning_strategy"]["name"] = args.lr_strategy
        params["l2_decay"] = args.l2_decay
        params["momentum_rate"] = args.momentum_rate
        optimizer = optimizer_setting(params)
        global_lr = optimizer._global_learning_rate()
        global_lr.persistable = True

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = 1
        exec_strategy.num_iteration_per_drop_scope = 30
        dist_strategy = DistributedStrategy()
        dist_strategy.exec_strategy = exec_strategy
        dist_strategy.enable_inplace = args.run_params['enable_inplace']
        dist_strategy.fuse_all_reduce_ops = args.run_params[
            'fuse_all_reduce_ops']
        dist_strategy.nccl_comm_num = args.run_params['nccl_comm_num']
        dist_strategy.use_local_sgd = args.run_params['use_local_sgd']
        dist_strategy.mode = args.run_params["mode"]
        dist_strategy.collective_mode = args.run_params["collective"]

        if args.run_params["fp16"]:
            optimizer = fluid.contrib.mixed_precision.decorate(
                optimizer,
                init_loss_scaling=128.0,
                use_dynamic_loss_scaling=True)

        if "use_dgc" in args.run_params and args.run_params["use_dgc"]:
            # use dgc must close fuse
            dist_strategy.fuse_all_reduce_ops = False
            optimizer = fluid.optimizer.DGCMomentumOptimizer(
                learning_rate=0.001, momentum=0.9, rampup_begin_step=0)

        dist_optimizer = fleet.distributed_optimizer(
            optimizer, strategy=dist_strategy)
        _, param_grads = dist_optimizer.minimize(self.avg_cost)

        shuffle_seed = 1
        train_reader = reader.train(
            settings=args, data_dir=DATA_DIR, pass_id_as_seed=shuffle_seed)
        self.py_reader.decorate_paddle_reader(
            paddle.batch(
                train_reader, batch_size=self.batch_size))

        if scale_loss > 1:
            avg_cost = fluid.layers.mean(x=cost) * scale_loss
        return self.avg_cost, self.py_reader

    def check_model_right(self, dirname):
        """
        check model.
        Args:
            dirname (str): model saved dirname
        """
        model_filename = os.path.join(dirname, "__model__")

        with open(model_filename, "rb") as f:
            program_desc_str = f.read()

        program = fluid.Program.parse_from_string(program_desc_str)
        with open(os.path.join(dirname, "__model__.proto"), "w") as wn:
            wn.write(str(program))

    def do_training(self, fleet, args=None):
        """
        begin training.
         Args:
            fleet (Collective): Collective inherited base class Fleet
            args (ArgumentParser): run args to config dist fleet.
        Returns:
            list: the return value is train loss.
        """
        train_fetch_vars = [self.avg_cost]
        train_fetch_list = []
        for var in train_fetch_vars:
            var.persistable = True
            train_fetch_list.append(var.name)

        trainer_prog = fleet._origin_program
        dist_prog = fleet.main_program

        device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        place = fluid.CUDAPlace(device_id)

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        train_info = []
        for pass_id in range(1):
            self.py_reader.start()
            batch_id = 0
            try:
                while True:
                    loss, = exe.run(dist_prog, fetch_list=train_fetch_list)
                    loss = np.mean(np.array(loss))
                    train_info.append(loss)
                    batch_id += 1
                    if batch_id == 5:
                        break
            except fluid.core.EOFException:
                self.py_reader.reset()
        return train_info


if __name__ == "__main__":
    runtime_main(TestDistResNet50)
