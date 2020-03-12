#!/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-12-09 20:39
# @Author  : liyang109

from __future__ import print_function

import nose.tools as tools
import os
from dist_base_fleet import TestFleetBase
from dist_base_fleet import run_by_freq


class TestDistResNet50(TestFleetBase):
    def __init__(self):
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.single_sync_gpu_data = [1.4922953, 1.4588202, 1.4054877, 1.3429222, 1.277166]
        self._model_file = 'dist_fleet_resnet.py'
        self.run_ce = os.getenv('RUN_CE', False)

    def check_data(self, loss, delta=None, expect=None):
        """
        校验结果数据
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
        """test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mc_cl"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': False, 'num_threads': 1, 'slice_var_up': True,
                           'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "local_sgd"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Fsvp_Tei_nn2_Tlsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp32_Fsvp_Tei_nn2_Tlsgd_mc_cl"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': False, 'num_threads': 1, 'slice_var_up': False,
                           'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "local_sgd"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Tlsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Tlsgd_mc_cl"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': False, 'num_threads': 1, 'slice_var_up': True,
                           'enable_inplace': False, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "local_sgd"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Fsvp_Fei_nn2_Tlsgd_mn_cl(self):
        """test_1tr_2gpu_nccl_fp32_Fsvp_Fei_nn2_Tlsgd_mn_cl"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': False, 'num_threads': 1, 'slice_var_up': False,
                           'enable_inplace': False, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "local_sgd"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mc_cg(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mc_cg"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': False, 'num_threads': 1, 'slice_var_up': True,
                           'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "grad_allreduce"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Fsvp_Tei_nn2_Tlsgd_mn_cg(self):
        """test_1tr_2gpu_nccl_fp32_Fsvp_Tei_nn2_Tlsgd_mn_cg"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': False, 'num_threads': 1, 'slice_var_up': False,
                           'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "grad_allreduce"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Tlsgd_mn_cg(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Tlsgd_mn_cg"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': False, 'num_threads': 1, 'slice_var_up': True,
                           'enable_inplace': False, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "grad_allreduce"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Fsvp_Fei_nn2_Tlsgd_mn_cg(self):
        """test_1tr_2gpu_nccl_fp32_Fsvp_Fei_nn2_Tlsgd_mn_cg"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': False, 'num_threads': 1, 'slice_var_up': False,
                           'enable_inplace': False, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "grad_allreduce"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Flsgd_mc_cg(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Flsgd_mc_cg"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': False, 'num_threads': 1, 'slice_var_up': True,
                           'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': False, 'mode': 'collective', 'collective': "grad_allreduce"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Fsvp_Tei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp32_Fsvp_Tei_nn2_Flsgd_mn_cc"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': False, 'num_threads': 1, 'slice_var_up': False,
                           'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': False, 'mode': 'nccl2', 'collective': None}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Flsgd_mn_cc"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': False, 'num_threads': 1, 'slice_var_up': True,
                           'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': False, 'mode': 'nccl2', 'collective': None}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Fsvp_Fei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp32_Fsvp_Fei_nn2_Flsgd_mn_cc"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': False, 'num_threads': 1, 'slice_var_up': False,
                           'enable_inplace': False, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': False, 'mode': 'nccl2', 'collective': None}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Fei_nn2_Flsgd_mn_cc"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': False, 'num_threads': 1, 'slice_var_up': True,
                           'enable_inplace': False, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': False, 'mode': 'nccl2', 'collective': None}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    """ fp16 """
    def test_1tr_2gpu_nccl_fp16_Tsvp_Tei_nn2_Tlsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp16_Tsvp_Tei_nn2_Tlsgd_mc_cl"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': True, 'num_threads': 1, 'slice_var_up': True,
                           'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "local_sgd"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    @run_by_freq(freq="MONTH")
    def test_1tr_2gpu_nccl_fp16_Fsvp_Tei_nn2_Tlsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp16_Fsvp_Tei_nn2_Tlsgd_mc_cl"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': True, 'num_threads': 1, 'slice_var_up': False,
                           'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "local_sgd"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    @run_by_freq(freq="MONTH")
    def test_1tr_2gpu_nccl_fp16_Tsvp_Fei_nn2_Tlsgd_mc_cl(self):
        """test_1tr_2gpu_nccl_fp16_Tsvp_Fei_nn2_Tlsgd_mc_cl"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': True, 'num_threads': 1, 'slice_var_up': True,
                           'enable_inplace': False, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "local_sgd"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    @run_by_freq(freq="MONTH")
    def test_1tr_2gpu_nccl_fp16_Fsvp_Fei_nn2_Tlsgd_mn_cl(self):
        """test_1tr_2gpu_nccl_fp16_Fsvp_Fei_nn2_Tlsgd_mn_cl"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': True, 'num_threads': 1, 'slice_var_up': False,
                           'enable_inplace': False, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "local_sgd"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Tsvp_Tei_nn2_Tlsgd_mc_cg(self):
        """test_1tr_2gpu_nccl_fp16_Tsvp_Tei_nn2_Tlsgd_mc_cg"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': True, 'num_threads': 1, 'slice_var_up': True,
                           'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "grad_allreduce"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    @run_by_freq(freq="MONTH")
    def test_1tr_2gpu_nccl_fp16_Fsvp_Tei_nn2_Tlsgd_mn_cg(self):
        """test_1tr_2gpu_nccl_fp16_Fsvp_Tei_nn2_Tlsgd_mn_cg"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': True, 'num_threads': 1, 'slice_var_up': False,
                           'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "grad_allreduce"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    @run_by_freq(freq="MONTH")
    def test_1tr_2gpu_nccl_fp16_Tsvp_Fei_nn2_Tlsgd_mn_cg(self):
        """test_1tr_2gpu_nccl_fp16_Tsvp_Fei_nn2_Tlsgd_mn_cg"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': True, 'num_threads': 1, 'slice_var_up': True,
                           'enable_inplace': False, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "grad_allreduce"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    @run_by_freq(freq="MONTH")
    def test_1tr_2gpu_nccl_fp16_Fsvp_Fei_nn2_Tlsgd_mn_cg(self):
        """test_1tr_2gpu_nccl_fp16_Fsvp_Fei_nn2_Tlsgd_mn_cg"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': True, 'num_threads': 1, 'slice_var_up': False,
                           'enable_inplace': False, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': True, 'mode': 'collective', 'collective': "grad_allreduce"}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

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

    @run_by_freq(freq="MONTH")
    def test_1tr_2gpu_nccl_fp16_Fsvp_Tei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp16_Fsvp_Tei_nn2_Flsgd_mn_cc"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': True, 'num_threads': 1, 'slice_var_up': False,
                           'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': False, 'mode': 'nccl2', 'collective': None}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Tsvp_Tei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp16_Tsvp_Tei_nn2_Flsgd_mn_cc"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': True, 'num_threads': 1, 'slice_var_up': True,
                           'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': False, 'mode': 'nccl2', 'collective': None}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Fsvp_Fei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp16_Fsvp_Fei_nn2_Flsgd_mn_cc"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': True, 'num_threads': 1, 'slice_var_up': False,
                           'enable_inplace': False, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': False, 'mode': 'nccl2', 'collective': None}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)

    def test_1tr_2gpu_nccl_fp16_Tsvp_Fei_nn2_Flsgd_mn_cc(self):
        """test_1tr_2gpu_nccl_fp16_Tsvp_Fei_nn2_Flsgd_mn_cc"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = { 'fp16': True, 'num_threads': 1, 'slice_var_up': True,
                           'enable_inplace': False, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           'use_local_sgd': False, 'mode': 'nccl2', 'collective': None}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=1e-0, expect=train_data_list2[0])
        self.check_data(train_data_list1[1], delta=1e-0, expect=self.single_sync_gpu_data)
    
    """use_dgc"""
    def test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mn_cl_dgc(self):
        """test_1tr_2gpu_nccl_fp32_Tsvp_Tei_nn2_Tlsgd_mn_cl_dgc"""
        TestFleetBase.__init__(self, pservers=0, trainers=1)
        self.run_params = {'sync': True, 'async': False, 'fp16': False, 'num_threads': 1, 'slice_var_up': True,
                           'enable_inplace': True, 'fuse_all_reduce_ops': 1, 'nccl_comm_num': 1,
                           #'use_local_sgd': True, 'mode': 'collective', 'collective': "local_sgd",
                           'use_local_sgd': False, 'mode': 'nccl2', 'collective': None,
                           'use_dgc': True}
        train_data_list1 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        train_data_list2 = self.get_result(self._model_file, update_method='nccl', gpu_num=2)
        assert len(train_data_list1) == 2
        assert len(train_data_list2) == 2
        self.check_data(train_data_list1[0], delta=5e-0, expect=train_data_list2[0])
