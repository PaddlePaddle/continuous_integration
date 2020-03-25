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
  * @file test_dist_fleet_save_persistables.py
  * @author liyang109@baidu.com
  * @date 2020-02-14
  * @brief
  *
  **************************************************************************/
"""
from __future__ import print_function
import nose.tools as tools
from dist_base_fleet import TestFleetBase
import os
import json
from dist_base_fleet import run_by_freq


class TestDistCTR(TestFleetBase):
    """Test dist save_persitable cases."""

    def __init__(self):
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.single_cpu_data = [
            2.6925561, 2.692213, 2.4876955, 1.225985, 2.522475
        ]
        self._model_file = 'dist_fleet_ctr.py'

    def check_data(self, loss, delta=None, expect=None):
        """
        校验结果数据.
        Args:
            loss (list): the loss will be checked.
            delta (float): 
            expect (list):
        """
        if expect:
            expect_data = expect
        else:
            expect_data = self.single_cpu_data
        if delta:
            for i in range(len(expect_data)):
                tools.assert_almost_equal(loss[i], expect_data[i], delta=delta)
        else:
            for i in range(len(expect_data)):
                tools.assert_equal(loss[i], expect_data[i])

    """sync"""

    def test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=1e-1, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-1, expect=self.single_cpu_data)
