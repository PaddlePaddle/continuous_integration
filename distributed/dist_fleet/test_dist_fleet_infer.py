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
  * @file test_dist_fleet_infer.py
  * @author liyang109@baidu.com
  * @date 2020-01-03 16:03
  * @brief 
  *
  **************************************************************************/
"""
from __future__ import print_function
import nose.tools as tools
import time
import signal
import os
import subprocess
import time


class TestDistInfer():
    """Test dist infer cases."""

    def __init__(self):
        self.single_data = [0.0069892979227006435]

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
            expect_data = self.single_data
        if delta:
            for i in range(len(expect_data)):
                tools.assert_almost_equal(loss[i], expect_data[i], delta=delta)
        else:
            for i in range(len(expect_data)):
                tools.assert_equal(loss[i], expect_data[i])

    def test_infer(self):
        """test infer process."""
        test_info1 = []
        test_info2 = []
        p1 = subprocess.Popen(
            "python dist_fleet_infer.py",
            shell=True,
            stderr=open("/tmp/infer1.log", "wb"),
            stdout=subprocess.PIPE)
        p1.wait()
        out, err = p1.communicate()
        lines = out.split("\n")[-2]
        if lines:
            lines = lines[1:-1].split(",")
        loss = [eval(i) for i in lines]
        test_info1.append(loss)
        time.sleep(2)
        p2 = subprocess.Popen(
            "python dist_fleet_infer.py",
            shell=True,
            stderr=open("/tmp/infer2.log", "wb"),
            stdout=subprocess.PIPE)
        out, err = p2.communicate()
        lines = out.split("\n")[-2]
        if lines:
            lines = lines[1:-1].split(",")
        loss = [eval(i) for i in lines]
        test_info2.append(loss)
        print('train_data1[0]:', test_info1[0][4])
        print('train_data2[0]:', test_info2[0][4])

        self.check_data(
            loss=test_info1[0][4], delta=1e-1, expect=test_info2[0][4])
        self.check_data(
            loss=test_info1[0][4], delta=1e-1, expect=self.single_data)
