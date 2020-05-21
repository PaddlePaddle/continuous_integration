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
@Author: liyang109
@Date: 2019/8/26 19:21
"""
from __future__ import print_function
import nose.tools as tools
from dist_base_fleet import TestFleetBase
import os
import json
from dist_base_fleet import run_by_freq
from dist_base_fleet import run_with_compatibility


class TestDistCTR(TestFleetBase):
    """Test dist ctr cases."""

    def __init__(self):
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.single_cpu_data = [
            2.5874448, 3.0397587, 3.0767221, 2.144417, 2.48193
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

    """async"""

    def test_ctr_1ps_1tr_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_1tr_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_1tr_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_1ps_1tr_async_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_1tr_async_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_1tr_async_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_1tr_async_1thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_1tr_async_1thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_1tr_async_1thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_1ps_1tr_async_1thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_1tr_async_1thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_1tr_async_1thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
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
        self.test_ctr_1ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_1ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_1ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_1tr_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_1tr_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_1tr_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_2ps_1tr_async_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_1tr_async_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_1tr_async_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_1tr_async_1thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_1tr_async_1thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_1tr_async_1thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_2ps_1tr_async_1thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_1tr_async_1thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_1tr_async_1thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
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
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """.test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        #self.check_data(train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        # self.check_data(train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    #def test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(self):
    #    """
    #    test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25
    #    """
    #    TestFleetBase.__init__(self, pservers=2, trainers=2)
    #    self.run_params = {'sync_mode':'async','cpu_num':2,'num_threads':2,
    #                       'slice_var_up':True,'enable_dc_asgd':False, 'split_method':True,
    #                       'runtime_split_send_recv':True,'geo_sgd':True,'wait_port':True,
    #                       'use_hierarchical_allreduce':True,'push_nums':25}
    #    self.test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(self.run_params)
    #    train_data_list1 = self.get_result(self._model_file, update_method='pserver')
    #    train_data_list2 = self.get_result(self._model_file, update_method='pserver')

    #    # 判断两个list输出是否为2
    #    assert len(train_data_list1) == 2
    #    assert len(train_data_list2) == 2
    #    #  两个train的loss值存在微小差距
    #    self.check_data(train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
    #    #loss值与预期相符
    #    self.check_data(train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_async_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        #loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    """geo_async"""

    def test_ctr_1ps_1tr_geo_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_1tr_geo_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_1tr_geo_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_1ps_1tr_geo_async_1thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_1tr_geo_async_1thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_1tr_geo_async_1thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
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
        self.test_ctr_1ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_1ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_1tr_geo_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_1tr_geo_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_1tr_geo_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_2ps_1tr_geo_async_1thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_1tr_geo_async_1thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_1tr_geo_async_1thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
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
        self.test_ctr_2ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_geo_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
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
        self.test_ctr_2ps_2tr_geo_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_geo_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_geo_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_geo_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_geo_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_geo_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_geo_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_geo_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_geo_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_geo_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_geo_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_geo_async_2thread_Fslice_Fdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_geo_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_geo_async_2thread_Tslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_geo_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_geo_async_2thread_Fslice_Tdc_Fsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_geo_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_geo_async_2thread_Tslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_2ps_2tr_geo_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_geo_async_2thread_Fslice_Fdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_geo_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_geo_async_2thread_Tslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_2ps_2tr_geo_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_geo_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'geo_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_geo_async_2thread_Fslice_Tdc_Tsm_Tsr_Tgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    """hsync"""

    def test_ctr_1ps_1tr_hasync_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_1tr_hasync_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_1tr_hasync_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_1tr_hasync_1thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_1tr_hasync_1thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_1tr_hasync_1thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_1tr_hasync_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_1tr_hasync_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_1tr_hasync_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_1tr_hasync_1thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_1tr_hasync_1thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_1tr_hasync_1thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_2ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_hasync_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_hasync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_hasync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_hasync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_hasync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_hasync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_hasync_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Tslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_hasync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_hasync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_hasync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_hasync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_hasync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_hasync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'half_async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_hasync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=3e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=3e-0, expect=self.single_cpu_data)

    """sync"""

    def test_ctr_1ps_1tr_sync_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_1tr_sync_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_1tr_sync_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_1ps_1tr_sync_1thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_1tr_sync_1thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': False,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_1tr_sync_1thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=self.single_cpu_data)

    def test_ctr_1ps_2tr_sync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_2tr_sync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_2tr_sync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_1ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': False,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_1tr_sync_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_1tr_sync_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_1tr_sync_1thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_2ps_1tr_sync_1thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_1tr_sync_1thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': False,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_1tr_sync_1thread_Fslice_Fdc_Fsm_Fsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        #  两个train的loss值存在微小差距
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_sync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_sync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_sync_2thread_Tslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_sync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_sync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_sync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_cpu_data)

    #def test_ctr_2ps_2tr_sync_2thread_Tslice_Fdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25(self):
    #    """
    #    test_ctr_2ps_2tr_sync_2thread_Tslice_Fdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25
    #    """
    #    TestFleetBase.__init__(self, pservers=2, trainers=2)
    #    self.run_params = {'sync_mode': 'sync', 'cpu_num': 2, 'num_threads': 2,
    #                       'slice_var_up': True, 'enable_dc_asgd': False, 'split_method': True,
    #                       'runtime_split_send_recv': False, 'geo_sgd': False, 'wait_port': True,
    #                       'use_hierarchical_allreduce': False, 'push_nums': 25}
    #    self.test_ctr_2ps_2tr_sync_2thread_Tslice_Fdc_Tsm_Fsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(self.run_params)
    #    train_data_list1 = self.get_result(self._model_file, update_method='pserver')
    #    train_data_list2 = self.get_result(self._model_file, update_method='pserver')

    #    # 判断两个list输出是否为2
    #    assert len(train_data_list1) == 2
    #    assert len(train_data_list2) == 2
    #    #  两个train的loss值存在微小差距
    #    self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
    #    # loss值与预期相符
    #    self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_sync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_sync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_sync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_sync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_sync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_sync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_sync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_sync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_sync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_sync_2thread_Fslice_Fdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_sync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_sync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_sync_2thread_Fslice_Tdc_Fsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_sync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_sync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_sync_2thread_Tslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_cpu_data)

    def test_ctr_2ps_2tr_sync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_sync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_sync_2thread_Fslice_Fdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_cpu_data)

    @run_by_freq(freq="MONTH")
    def test_ctr_2ps_2tr_sync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_sync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_sync_2thread_Tslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_cpu_data)

    @run_with_compatibility
    def test_ctr_2ps_2tr_sync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25(
            self):
        """test_ctr_2ps_2tr_sync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'sync_mode': 'sync',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_dc_asgd': False,
            'split_method': True,
            'runtime_split_send_recv': False,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': True,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_sync_2thread_Fslice_Tdc_Tsm_Tsr_Fgeo_Twp_Tha_pn25.__func__.__doc__ = json.dumps(
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
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        # loss值与预期相符
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_cpu_data)

    """DataSet"""

    def test_ctr_1ps_1tr_dataset_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_1tr_dataset_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'run_from_dataset': True,
            'sync_mode': 'async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_1tr_dataset_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        assert len(train_data_list1[0]) > 0

    def test_ctr_1ps_2tr_dataset_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_2tr_dataset_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'run_from_dataset': True,
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_2tr_dataset_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        assert len(train_data_list1[0]) > 0

    def test_ctr_2ps_1tr_dataset_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_1tr_dataset_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'run_from_dataset': True,
            'sync_mode': 'async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_1tr_dataset_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        assert len(train_data_list1[0]) > 0

    def test_ctr_1ps_1tr_dataset_geo_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_1tr_dataset_geo_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'run_from_dataset': True,
            'sync_mode': 'geo_async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_1ps_1tr_dataset_geo_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        assert len(train_data_list1[0]) > 0

    def test_ctr_1ps_2tr_dataset_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_2tr_dataset_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'run_from_dataset': True,
            'sync_mode': 'geo_async',
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
        self.test_ctr_1ps_2tr_dataset_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        assert len(train_data_list1[0]) > 0

    def test_ctr_2ps_1tr_dataset_geo_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_1tr_dataset_geo_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'run_from_dataset': True,
            'sync_mode': 'geo_async',
            'cpu_num': 1,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': True,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_1tr_dataset_geo_async_1thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        assert len(train_data_list1[0]) > 0

    def test_ctr_2ps_2tr_dataset_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_dataset_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'run_from_dataset': True,
            'sync_mode': 'async',
            'cpu_num': 2,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_dc_asgd': False,
            'split_method': False,
            'runtime_split_send_recv': True,
            'geo_sgd': False,
            'wait_port': True,
            'use_hierarchical_allreduce': False,
            'push_nums': 25
        }
        self.test_ctr_2ps_2tr_dataset_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        assert len(train_data_list1[1]) > 0

    def test_ctr_1ps_1tr_dataset_half_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_1tr_dataset_half_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=1)
        self.run_params = {
            'run_from_dataset': True,
            'sync_mode': 'half_async',
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
        self.test_ctr_1ps_1tr_dataset_half_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        assert len(train_data_list1[0]) > 0

    def test_ctr_1ps_2tr_dataset_half_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_2tr_dataset_half_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'run_from_dataset': True,
            'sync_mode': 'half_async',
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
        self.test_ctr_1ps_2tr_dataset_half_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        assert len(train_data_list1[1]) > 0

    def test_ctr_2ps_1tr_dataset_half_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_1tr_dataset_half_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'run_from_dataset': True,
            'sync_mode': 'half_async',
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
        self.test_ctr_2ps_1tr_dataset_half_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        assert len(train_data_list1[0]) > 0

    def test_ctr_2ps_2tr_dataset_half_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_dataset_half_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'run_from_dataset': True,
            'sync_mode': 'half_async',
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
        self.test_ctr_2ps_2tr_dataset_half_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        assert len(train_data_list1[1]) > 0

    # def test_ctr_1ps_1tr_dataset_sync_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
    #         self):
    #     """test_ctr_1ps_1tr_dataset_sync_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
    #     TestFleetBase.__init__(self, pservers=1, trainers=1)
    #     self.run_params = {
    #         'run_from_dataset': True,
    #         'sync_mode': 'sync',
    #         'cpu_num': 2,
    #         'num_threads': 2,
    #         'slice_var_up': True,
    #         'enable_dc_asgd': False,
    #         'split_method': False,
    #         'runtime_split_send_recv': True,
    #         'geo_sgd': True,
    #         'wait_port': True,
    #         'use_hierarchical_allreduce': False,
    #         'push_nums': 25
    #     }
    #     self.test_ctr_1ps_1tr_dataset_sync_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
    #         self.run_params)
    #     train_data_list1 = self.get_result(
    #         self._model_file, update_method='pserver')
    #     train_data_list2 = self.get_result(
    #         self._model_file, update_method='pserver')

    #     # 判断两个list输出是否为2
    #     assert len(train_data_list1) == 1
    #     assert len(train_data_list2) == 1
    #     assert len(train_data_list1[0]) > 0

    def test_ctr_1ps_2tr_dataset_sync_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_1ps_2tr_dataset_sync_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=1, trainers=2)
        self.run_params = {
            'run_from_dataset': True,
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
        self.test_ctr_1ps_2tr_dataset_sync_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        assert len(train_data_list1[1]) > 0

    def test_ctr_2ps_1tr_dataset_sync_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_1tr_dataset_sync_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=1)
        self.run_params = {
            'run_from_dataset': True,
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
        self.test_ctr_2ps_1tr_dataset_sync_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 1
        assert len(train_data_list2) == 1
        assert len(train_data_list1[0]) > 0

    def test_ctr_2ps_2tr_dataset_sync_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_dataset_sync_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'run_from_dataset': True,
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
        self.test_ctr_2ps_2tr_dataset_sync_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        assert len(train_data_list1[0]) > 0

    def test_ctr_2ps_2tr_dataset_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25(
            self):
        """test_ctr_2ps_2tr_dataset_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25."""
        TestFleetBase.__init__(self, pservers=2, trainers=2)
        self.run_params = {
            'run_from_dataset': True,
            'sync_mode': 'geo_async',
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
        self.test_ctr_2ps_2tr_dataset_geo_async_2thread_Tslice_Fdc_Fsm_Tsr_Tgeo_Twp_Fha_pn25.__func__.__doc__ = json.dumps(
            self.run_params)
        train_data_list1 = self.get_result(
            self._model_file, update_method='pserver')
        train_data_list2 = self.get_result(
            self._model_file, update_method='pserver')

        # 判断两个list输出是否为2
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        assert len(train_data_list1[1]) > 0
