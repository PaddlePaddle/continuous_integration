#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
run case script
Authors: xuguangyao01(xuguangyao01@baidu.com)
Date:    2019/03/19 14:05:14
"""

import os
import sys
import json


def run_all(batch):
    """
    run all cases under the directory of 'case'
    """
    suite_list = os.listdir("../ce/{}/case".format(batch))
    success_op_list = []
    failure_op_list = []
    counter = 0
    for suite_name in sorted(suite_list):
        counter += 1
        with open(os.path.join("../ce/{}/case".format(batch), suite_name), "r") as fin:
            try:
                suite = json.load(fin)
            except Exception:
                print suite_name
                sys.exit(1)
        print "BEGIN OP TEST: {}".format(suite_name)
        for case_name in suite:
            print "BEGIN CASE: {}".format(case_name)
            cmd = "python paddle_op_test.py '{0}' {1} {2} ../ce/{3}/result > ../ce/{3}/log/{1}.{2} 2>&1"\
                  .format(json.dumps(suite[case_name]), case_name, test_type, batch)
            if os.system(cmd) != 0:
                failure_op_list.append(case_name)
            else:
                success_op_list.append(case_name)
        print ""
    print "{} success {} failure".format(len(success_op_list), len(failure_op_list))
    for op in failure_op_list:
        print "[f] " + op


def run_suite(batch, suite_pattern):
    """
    run the given suite
    """
    suite_list = os.listdir("../ce/{}/case".format(batch))
    for suite_name in sorted(suite_list):
        if suite_pattern not in suite_name:
            continue
        with open(os.path.join("../ce/{}/case".format(batch), suite_name), "r") as fin:
            suite = json.load(fin)
            print "BEGIN OP TEST: {}".format(suite_name)
            for case_name in suite:
                print "BEGIN CASE: {}".format(case_name)
                cmd = "python paddle_op_test.py '{0}' {1} {2} ../ce/{3}/result > ../ce/{3}/log/{1}.{2} 2>&1"\
                      .format(json.dumps(suite[case_name]), case_name, test_type, batch)
                os.system(cmd)
            print ""


def run_case(batch, suite_name, case_name):
    """
    run the given case
    """
    with open(os.path.join("../ce/{}/case".format(batch), suite_name), "r") as fin:
        suite = json.load(fin)
        print "BEGIN OP TEST: {}".format(suite_name)
        print "BEGIN CASE: {}".format(case_name)
        cmd = "python paddle_op_test.py '{0}' {1} {2} ../ce/{3}/result > ../ce/{3}/log/{1}.{2} 2>&1"\
              .format(json.dumps(suite[case_name]), case_name, test_type, batch)
        os.system(cmd)


if __name__ == "__main__":
    test_type = sys.argv[1]
    param_len = len(sys.argv[1:])
    if param_len == 2:
        run_all(sys.argv[2])
    elif param_len == 3:
        run_suite(sys.argv[2], sys.argv[3])
    elif param_len == 4:
        run_case(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise Exception("param error")
