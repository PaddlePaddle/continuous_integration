#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
#======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
#======================================================================
"""
@Desc: test_dist_fleet_vgg module
@File: test_dist_fleet_vgg.py
@Author: liangjinhua
@Date: 2019/8/26 19:21
"""
from __future__ import print_function
import nose.tools as tools
import os
from .dist_base_fleet import TestFleetBase
from .dist_base_fleet import run_by_freq
import json


class TestDistVgg16(TestFleetBase):
    """VGG test cases."""

    def __init__(self):
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.single_sync_gpu_data = [
            1.5387154, 1.5342727, 1.525869, 1.5140724, 1.499335
        ]
        self._model_file = 'dist_fleet_vgg.py'

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
            expect_data = self.single_sync_gpu_data
        if delta:
            for i in range(len(expect_data)):
                tools.assert_almost_equal(loss[i], expect_data[i], delta=delta)
        else:
            for i in range(len(expect_data)):
                tools.assert_equal(loss[i], expect_data[i])

    """ FP32 """

    def test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mc_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "local_sgd"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Fsvp_Tei_nn2_Tlsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp32_Fsvp_Tei_nn2_Tlsgd_mc_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "local_sgd"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Tlsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Tlsgd_mc_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "local_sgd"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Fsvp_Fei_nn2_Tlsgd_mn_cl(self):
        """test_1tr_2gpu_nccl_fp32_Fsvp_Fei_nn2_Tlsgd_mn_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "local_sgd"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mc_cg(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mc_cg."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "grad_allreduce"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Fsvp_Tei_nn2_Tlsgd_mn_cg(self):
        """test_1tr_2gpu_nccl_fp32_Fsvp_Tei_nn2_Tlsgd_mn_cg."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "grad_allreduce"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Tlsgd_mn_cg(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Tlsgd_mn_cg."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "grad_allreduce"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Fsvp_Fei_nn2_Tlsgd_mn_cg(self):
        """test_1tr_2gpu_nccl_fp32_Fsvp_Fei_nn2_Tlsgd_mn_cg."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "grad_allreduce"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Flsgd_mc_cg(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Flsgd_mc_cg."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'collective',
            'collective': "grad_allreduce"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Fsvp_Tei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp32_Fsvp_Tei_nn2_Flsgd_mn_cc."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': None
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Flsgd_mn_cc."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': None
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Fsvp_Fei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp32_Fsvp_Fei_nn2_Flsgd_mn_cc."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': None
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Flsgd_mn_cc."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': None
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_2tr_1gpu_nccl_fp32_Tsvp_Tei_nn1_Tlsgd_mc_cl(self):
        """test_2tr_1gpu_nccl_fp32_Tsvp_Tei_nn1_Tlsgd_mc_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=2)
        self.run_params = {
            'fp16': False,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "local_sgd"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=1)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=1)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_2tr_2gpu_nccl_fp32_Tsvp_Tei_nn1_Tlsgd_mc_cl(self):
        """test_2tr_2gpu_nccl_fp32_Tsvp_Tei_nn1_Tlsgd_mc_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=2)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "local_sgd"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 4
        assert len(train_data_list2) == 4
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])

    """ fp16 """

    def test_1tr_2gpu_nccl_fp16_Tsvp_Tei_nn2_Tlsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp16_Tsvp_Tei_nn2_Tlsgd_mc_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': True,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "local_sgd"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Fsvp_Tei_nn2_Tlsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp16_Fsvp_Tei_nn2_Tlsgd_mc_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': True,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "local_sgd"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Tsvp_Fei_nn2_Tlsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp16_Tsvp_Fei_nn2_Tlsgd_mc_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': True,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "local_sgd"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Fsvp_Fei_nn2_Tlsgd_mn_cl(self):
        """test_1tr_2gpu_nccl_fp16_Fsvp_Fei_nn2_Tlsgd_mn_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': True,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "local_sgd"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Tsvp_Tei_nn2_Tlsgd_mc_cg(self):
        """test_1tr_2gpu_nccl_fp16_Tsvp_Tei_nn2_Tlsgd_mc_cg."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': True,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "grad_allreduce"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Fsvp_Tei_nn2_Tlsgd_mn_cg(self):
        """test_1tr_2gpu_nccl_fp16_Fsvp_Tei_nn2_Tlsgd_mn_cg."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': True,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "grad_allreduce"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Tsvp_Fei_nn2_Tlsgd_mn_cg(self):
        """test_1tr_2gpu_nccl_fp16_Tsvp_Fei_nn2_Tlsgd_mn_cg."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': True,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "grad_allreduce"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Fsvp_Fei_nn2_Tlsgd_mn_cg(self):
        """test_1tr_2gpu_nccl_fp16_Fsvp_Fei_nn2_Tlsgd_mn_cg."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': True,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "grad_allreduce"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    # def test_1tr_2gpu_nccl_fp16_Tsvp_Tei_nn2_Flsgd_mc_cg(self):
    #     """test_1tr_2gpu_nccl_fp16_Tsvp_Tei_nn2_Flsgd_mc_cg"""
    #     TestFleetBase.__init__(self, pservers=0, trainers=1)
    #     self.run_params = { 'fp16': True, 'num_threads': 1, 'slice_var_up': True,
    #                        'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
    #                        'use_local_sgd': False, 'mode': 'collective', 'collective': "grad_allreduce"}
    #     train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
    #     train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
    #     assert len(train_data_list1) == 2
    #     assert len(train_data_list2) == 2
    #     self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
    #     self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Fsvp_Tei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp16_Fsvp_Tei_nn2_Flsgd_mn_cc."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': True,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': None
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Tsvp_Tei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp16_Tsvp_Tei_nn2_Flsgd_mn_cc."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': True,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': None
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Fsvp_Fei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp16_Fsvp_Fei_nn2_Flsgd_mn_cc."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': True,
            'num_threads': 1,
            'slice_var_up': False,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': None
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Tsvp_Fei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp16_Tsvp_Fei_nn2_Flsgd_mn_cc."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': True,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': None
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Flsgd_mn_cc_fs_tr(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Flsgd_mn_cc_fs_tr."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': None,
            'sequential': False,
            'recompute': True
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Flsgd_mn_cc_ts_fr(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Flsgd_mn_cc_ts_fr."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': None,
            'sequential': True,
            'recompute': False
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Flsgd_mn_cc_fs_fr(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Flsgd_mn_cc_fs_fr."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': False,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': None,
            'sequential': False,
            'recompute': False
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_2tr_1gpu_nccl_fp16_Tsvp_Tei_nn2_Tlsgd_mc_cl(self):
        """test_2tr_1gpu_nccl_fp16_Tsvp_Tei_nn2_Tlsgd_mc_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=2)
        self.run_params = {
            'fp16': True,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "local_sgd"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=1)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=1)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(
            train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_2tr_2gpu_nccl_fp16_Fsvp_Tei_nn2_Tlsgd_mc_cl(self):
        """test_2tr_2gpu_nccl_fp16_Fsvp_Tei_nn2_Tlsgd_mc_cl."""
        TestFleetBase.__init__(self, pservers=0, trainers=2)
        self.run_params = {
            'fp16': True,
            'num_threads': 2,
            'slice_var_up': False,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            'use_local_sgd': True,
            'mode': 'collective',
            'collective': "local_sgd"
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 4
        assert len(train_data_list2) == 4
        self.check_data(
            train_data_list1[0], delta=1e-0, expect=train_data_list2[0])

    """use_dgc"""

    def test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mn_cl_dgc(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mn_cl_dgc."""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {
            'sync': True,
            'async': False,
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            #'use_local_sgd': True, 'mode': 'collective', 'collective': "local_sgd",
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': None,
            'use_dgc': True
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=5e-0, expect=train_data_list2[0])

    def test_2tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mn_cl_dgc(self):
        """test_2tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mn_cl_dgc."""
        TestFleetBase.__init__(self, pservers=0, trainers=2)
        self.run_params = {
            'sync': True,
            'async': False,
            'fp16': False,
            'num_threads': 2,
            'slice_var_up': True,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            #'use_local_sgd': True, 'mode': 'collective', 'collective': "local_sgd",
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': None,
            'use_dgc': True
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 4
        assert len(train_data_list2) == 4
        self.check_data(
            train_data_list1[0], delta=5e-0, expect=train_data_list2[0])

    def test_2tr_1gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mn_cl_dgc(self):
        """test_2tr_1gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mn_cl_dgc."""
        TestFleetBase.__init__(self, pservers=0, trainers=2)
        self.run_params = {
            'sync': True,
            'async': False,
            'fp16': False,
            'num_threads': 1,
            'slice_var_up': True,
            'enable_inplace': True,
            'fuse_all_reduce_ops': 1,
            'nccl_comm_num': 1,
            #'use_local_sgd': True, 'mode': 'collective', 'collective': "local_sgd",
            'use_local_sgd': False,
            'mode': 'nccl2',
            'collective': None,
            'use_dgc': True
        }
        train_data_list1 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=1)
        train_data_list2 = self.get_result(
            self._model_file, update_method='nccl', gpu_num=1)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(
            train_data_list1[0], delta=5e-0, expect=train_data_list2[0])
