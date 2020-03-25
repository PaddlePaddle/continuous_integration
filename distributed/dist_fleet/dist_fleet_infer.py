#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
#======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
#======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file dist_fleet_infer.py
  * @author liyang109@baidu.com
  * @date 2020-01-03 16:00
  * @brief 
  *
  **************************************************************************/
"""
from __future__ import print_function
import os
import time
import numpy as np
import logging
import paddle
import paddle.fluid as fluid
import math
import sys
sys.path.append('./thirdparty/ctr')
import py_reader_generator as py_reader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

DATA_PATH = 'thirdparty/data/dist_data/ctr_data/part-100'


def input_data():
    """
    input data for ctr model.
    Returns:
        list: The return value contains dense_input,sparse_input, label.
    """
    dense_input = fluid.layers.data(
        name="dense_input", shape=[13], dtype="float32")

    sparse_input_ids = [
        fluid.layers.data(
            name="C" + str(i), shape=[1], lod_level=1, dtype="int64")
        for i in range(1, 27)
    ]

    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    inputs = [dense_input] + sparse_input_ids + [label]
    return inputs


def net():
    """
    ctr net struct.
    Returns:
        tuple: the return value contains avg_cost, auc_var, batch_auc_var.
    """

    def embedding_layer(input):
        """embedding layer is sparse."""
        return fluid.layers.embedding(
            input=input,
            is_sparse=True,
            size=[1000001, 10],
            param_attr=fluid.ParamAttr(
                name="SparseFeatFactors",
                initializer=fluid.initializer.Uniform()), )

    inputs = input_data()
    sparse_embed_seq = list(map(embedding_layer, inputs[1:-1]))
    concated = fluid.layers.concat(sparse_embed_seq + inputs[0:1], axis=1)

    fc1 = fluid.layers.fc(
        input=concated,
        size=400,
        act="relu",
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
            scale=1 / math.sqrt(concated.shape[1]))), )
    fc2 = fluid.layers.fc(
        input=fc1,
        size=400,
        act="relu",
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
            scale=1 / math.sqrt(fc1.shape[1]))), )
    fc3 = fluid.layers.fc(
        input=fc2,
        size=400,
        act="relu",
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
            scale=1 / math.sqrt(fc2.shape[1]))), )
    predict = fluid.layers.fc(
        input=fc3,
        size=2,
        act="softmax",
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
            scale=1 / math.sqrt(fc3.shape[1]))), )

    cost = fluid.layers.cross_entropy(input=predict, label=input_data()[-1])
    avg_cost = fluid.layers.reduce_sum(cost)
    auc_var, batch_auc_var, _ = fluid.layers.auc(input=predict,
                                                 label=input_data()[-1],
                                                 num_thresholds=2**12,
                                                 slide_steps=20)
    return avg_cost, auc_var, batch_auc_var


def run_infer(model_path):
    """
    run infer from existed model file.
    Args:
        model_path (str): model save path
    Returns:
        list: the losses of trainer
    """
    place = fluid.CPUPlace()
    train_generator = py_reader.CriteoDataset(1000001)
    file_list = [str(DATA_PATH)] * 2
    test_reader = paddle.batch(train_generator.test(file_list), batch_size=4)
    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()

    def set_zero(var_name):
        """auc set zero."""
        param = fluid.global_scope().var(var_name).get_tensor()
        param_array = np.zeros(param._get_dims()).astype("int64")
        param.set(param_array, place)

    with fluid.framework.program_guard(test_program, startup_program):
        with fluid.unique_name.guard():
            inputs = input_data()
            # inputs = ctr_model.input_data(params)
            loss, auc_var, batch_auc_var = net()

            exe = fluid.Executor(place)
            feeder = fluid.DataFeeder(feed_list=inputs, place=place)

            fluid.io.load_persistables(
                executor=exe,
                dirname=model_path,
                main_program=fluid.default_main_program())

            auc_states_names = [
                '_generated_var_0', '_generated_var_1', '_generated_var_2',
                '_generated_var_3'
            ]
            for name in auc_states_names:
                set_zero(name)

            run_index = 0
            infer_auc = 0
            L = []
            test_info = []
            for batch_id, data in enumerate(test_reader()):
                loss_val, auc_val = exe.run(test_program,
                                            feed=feeder.feed(data),
                                            fetch_list=[loss, auc_var])
                run_index += 1
                infer_auc = auc_val
                L.append(loss_val / 250)
                if batch_id % 1 == 0:
                    logger.info("TEST --> batch: {} loss: {} auc: {}".format(
                        batch_id, loss_val / 250, auc_val))
                    loss_val = loss_val / 250
                    test_info.append(loss_val.tolist())
                if batch_id == 5:
                    break
            infer_loss = np.mean(L)
            infer_result = {}
            infer_result['loss'] = infer_loss
            infer_result['auc'] = infer_auc
            log_path = model_path + '/infer_result.log'
            logger.info(str(infer_result))
            with open(log_path, 'w+') as f:
                f.write(str(infer_result))
            logger.info("Inference complete")
    return test_info


if __name__ == "__main__":
    model_list = []
    model_path = './dist_model_ctr'
    for _, dir, _ in os.walk(model_path):
        for model in dir:
            if "fin" in model:
                path = "/".join([model_path, model])
                model_list.append(path)
    for model in model_list:
        logger.info("Test model {}".format(model))
        test_info = run_infer(model)
        print(test_info)
