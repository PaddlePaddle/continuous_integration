#!/usr/bin/env pythmn
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
paddle op test
Authors: xuguangyao01(xuguangyao01@baidu.com)
Date:    2019/03/19 14:50:10
"""


import sys
import unittest
import paddle.fluid as fluid
import paddle
import numpy
import math
import random
import json
import time
import pprint
import pynvml
import psutil
import os
import traceback
import tensorflow as tf
import lib
import re
import torch
from torch.autograd import Variable


class ParametrizedTestCase(unittest.TestCase):
    """ 
    TestCase classes that want to be parametrized should
    inherit from this class.
    """
    def __init__(self, methodName='runTest', param=None, test_type="fast"):
        super(ParametrizedTestCase, self).__init__(methodName)
        self.param = param

    @staticmethod
    def parametrize(testcase_class, param=None, test_type="fast"):
        """
        Create a suite containing all tests taken from the given 
        subclass, passing them the parameter 'param'. 
        """  
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_class)
        suite = unittest.TestSuite()
        for name in testnames:
            if test_type in name:
                suite.addTest(testcase_class(name, param=param))
        return suite


class PaddleTest(ParametrizedTestCase):
    """
    paddle op test class
    """
    def setUp(self):
        """
        parse param to json
        """
        self.param_json = json.loads(self.param)

    def get_memory(self, type=0):
        """
        get memory
        """
        memory = 0
        # gpu
        if type == 1:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ['CUDA_VISIBLE_DEVICES']))
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                memory = process.usedGpuMemory / 1024 / 1024          
            pynvml.nvmlShutdown()
            print "gpu memory", memory
        # cpu
        else:
            pid = os.getpid()
            process = psutil.Process(pid)
            memory = process.memory_info().rss / 1024 / 1024
        return memory

    def prepare(self, op, params, test_type="fast"):
        """
        op test
        """
        feed_key, feed_value, param_list, data_list = [], [], [], []
        tf_feed_key, tf_param_list, tf_data_list = [], [], []
        exec_dict = {"feed": [], "format": [], "paddle": [], "tf": []}
	
        # generate params
        for index, param in enumerate(params):
            if param["type"] == "variable":
                shape_list = []
                for i in range(param["dim"]):
                    if param["data_generator"][i]["type"] == "random":
                        shape_list.append(str(random.randint(*param["data_generator"][i]["range"])))
                    elif param["data_generator"][i]["type"] == "default":
                        shape_list.append(str(param["data_generator"][i]["value"]))
                    else:
                        raise(Exception("input error"))
                shape_str = ",".join(shape_list)
                feed_method = param.get("feed", "random.randn")
                if type(feed_method) in (str, unicode):
                    if "lod" in param:
                        cmd = "feed_var{} = fluid.create_lod_tensor(numpy.{}({}).astype('{}'), {}, {})"\
                              .format(index, feed_method, "[{}]".format(shape_str) if feed_method in \
                                      ("random.random", "ones", "zeros") else shape_str, param.get("dtype", "int32"), str(param["lod"]),
                                      "fluid.CUDAPlace(0)" if self.param_json.get("gpu", 0) else "fluid.CPUPlace()")
                    else:
                        cmd = "feed_var{} = numpy.{}({}).astype('{}')"\
                              .format(index, feed_method, "[{}]".format(shape_str) if feed_method in \
                                      ("random.random", "ones", "zeros") else shape_str, param.get("dtype", "float32"))
                    if "cross_entropy" in op and param["name"] == "label":
		    	feed_method = "random.randint"
                        cmd = "feed_var{} = numpy.{}({}).astype('{}')"\
                              .format(index, feed_method, "1,2,[{}]".format(shape_str) if feed_method in \
                                      ("random.random", "ones", "zeros", "random.randint") else shape_str, param.get("dtype", "float32"))
                    if "one_hot" in op and param["name"] == "input":
		    	feed_method = "random.randint"
                        cmd = "feed_var{} = numpy.{}({}).astype('{}')"\
                              .format(index, feed_method, "1,10,[{}]".format(shape_str) if feed_method in \
                                      ("random.random", "ones", "zeros", "random.randint") else shape_str, param.get("dtype", "float32"))
                # feed is a given list
                else:
                    if "lod" in param:
                        cmd = "feed_var{} = fluid.create_lod_tensor(numpy.array({}).astype('{}'), {}, {})"\
                              .format(index, feed_method, param.get("dtype", "int32"), str(param["lod"]),
                                      "fluid.CUDAPlace(0)" if self.param_json["gpu"] else "fluid.CPUPlace()")
                    else:
                        cmd = "feed_var{} = numpy.array({}).astype('{}')".format(index, str(feed_method), param.get("dtype", "int32"))
                    
                exec_dict["feed"].append(cmd)
                exec_dict["format"].append(param.get("is_format", 0))
                if "name" in param:
                    exec_dict["paddle"].append("data{0} = fluid.layers.data(name='data{0}', shape=[{1}], dtype='{2}'"\
                                               ",\\\n        append_batch_size=False, stop_gradient=False, lod_level={3})"\
                                               .format(index, shape_str, param.get("dtype", "float32"), param.get("lod_level", 0)))
                    data_list.append("data{}".format(index))
                    feed_key.append("data{}".format(index))
                    feed_value.append("feed_var{}".format(index))
                    if test_type == "fast":
                        param_list.append("{}=data_list[{}]".format(param["name"], len(data_list)-1))
                    else:
                        param_list.append("{}=fluid.dygraph.to_variable(feed_value[{}])".format(param["name"], len(data_list)-1))
                if "tf-name" in param:
                    exec_dict["tf"].append("tf_data{2} = tf.placeholder(dtype=tf.{0}, shape=[{1}], name='tf_data{2}')"\
                                           .format(param.get("dtype", "float32"), shape_str, index))
                    tf_data_list.append("tf_data{}".format(index))
                    tf_feed_key.append("tf_data{}".format(index))
                    tf_param_list.append("{}=tf_data_list[{}]".format(param["tf-name"], len(tf_data_list)-1))
            elif param["type"] in ("tuple", "list"):
                size_list = []
                for i in range(param["size"]):
                    if param["data_generator"][i]["type"] == "random":
                        size_list.append(str(random.randint(*param["data_generator"][i]["range"])))
                    elif param["data_generator"][i]["type"] == "default":
                        size_list.append(str(param["data_generator"][i]["value"]))
                    else:
                        raise(Exception("input error"))
                if param.get("is_tensor", 0):
                    tensor_num = random.randint(1, len(size_list))
                    i = 0
                    tensor_flag_list = []
                    while i < len(size_list):
                        if i < tensor_num:
                            tensor_flag_list.append(1)
                        else:
                            tensor_flag_list.append(0)
                        i += 1
                    random.shuffle(tensor_flag_list)
                    if "name" in param:
                        param_str = "[{}]".format(",".join(['fluid.layers.fill_constant([1], "{}", {})'.format(param.get("dtype", "int32"), value)\
                                    if tensor_flag_list[index] else value for index, value in enumerate(size_list)]))
                        param_list.append("{}={}".format(param["name"], param_str))
                    if "tf-name" in param:
                        param_str = "[{}]".format(",".join(['tf.constant({}, "{}", [])'.format(value, param.get("dtype", "int32"))\
                                    if tensor_flag_list[index] else value for index, value in enumerate(size_list)]))
                        tf_param_list.append("{}={}".format(param["tf-name"], param_str))
                else:
                    if "name" in param:
                        param_str = "[{}]".format(",".join(size_list))
                        param_list.append("{}={}".format(param["name"], param_str))
                    if "tf-name" in param:
                        param_str = "[{}]".format(",".join(size_list))
                        tf_param_list.append("{}={}".format(param["tf-name"], param_str))
            elif param["type"] == "string":
                dict_t = {"name": param_list, "tf-name": tf_param_list}
                if param["data_generator"]["type"] == "choice":
                    value = random.choice(param["data_generator"]["option"])
                elif param["data_generator"]["type"] == "default":
                    value = param["data_generator"]["value"]
                else:
                    raise(Exception("input error"))
                for key in dict_t:
                    if key not in param:
                        continue
                    if key == "tf-name" and param[key] == "dtype":
                        dict_t[key].append("{}={}".format(param[key], "tf." + value))
                    else:
                        dict_t[key].append("{}='{}'".format(param[key], value))
            elif param["type"] == "bool":
                dict_t = {"name": param_list, "tf-name": tf_param_list}
                if param["data_generator"]["type"] == "choice":
                    value = random.choice(param["data_generator"]["option"])
                elif param["data_generator"]["type"] == "default":
                    value = param["data_generator"]["value"]
                else:
                    raise(Exception("input error"))
                for key in dict_t:
                    if key not in param:
                        continue
                    dict_t[key].append("{}={}".format(param[key], value))
            elif param["type"] in ("float", "int"):
                dict_t = {"name": param_list, "tf-name": tf_param_list}
                if param["data_generator"]["type"] == "choice":
                    value = random.choice(param["data_generator"]["option"])
                elif param["data_generator"]["type"] == "random":
                    value = random.random() + random.randint(*param["data_generator"]["range"])\
                            if param["type"] == "float" else\
                            random.randint(*param["data_generator"]["range"])
                elif param["data_generator"]["type"] == "default":
                    value = param["data_generator"]["value"]
                else:
                    raise(Exception("input error"))
                for key in dict_t:
                    if key not in param:
                        continue
                    dict_t[key].append("{}={}".format(param[key], value))
            elif param["type"] == "param_attr":
                dict_t = {"name": param_list, "tf-name": tf_param_list}
                for key in dict_t:
                    if key not in param:
                        continue
                    param_attr = "fluid.ParamAttr({})".format(",".join(["{}={}".format(\
                                 attr_key, param["attribute"][attr_key]) for attr_key in param["attribute"]]))
                    dict_t[key].append("{}={}".format(param[key], param_attr))
            elif param["type"] == "None":
                dict_t = {"name": param_list, "tf-name": tf_param_list}
                for key in dict_t:
                    if key not in param:
                        continue
                    dict_t[key].append("{}=None".format(param[key]))
            else:
                raise(Exception("input error"))

        return {"feed_key": feed_key,
                "feed_value": feed_value,
                "param_list": param_list,
                "data_list": data_list,
                "tf_feed_key": tf_feed_key,
                "tf_param_list": tf_param_list,
                "tf_data_list": tf_data_list,
                "exec_dict": exec_dict}

    def test_op_slow(self):
        """
        op test
        """
        op_name = self.param_json["op"]
        ret_dict = self.prepare(op_name, self.param_json["params"])
        feed_key = ret_dict["feed_key"]
        feed_value = ret_dict["feed_value"]
        param_list = ret_dict["param_list"]
        data_list = ret_dict["data_list"]
        exec_dict = ret_dict["exec_dict"]

        for exec_cmd in exec_dict["feed"]:
            exec(exec_cmd)
        for exec_cmd in exec_dict["paddle"]:
            exec(exec_cmd)
        for i in range(len(data_list)):
            data_list[i] = eval(data_list[i])
        for i in range(len(feed_value)):
            feed_value[i] = eval(feed_value[i])

        # define op
        op_calc = "{}({})".format("fluid.layers." + op_name if "." not in op_name else op_name, ",".join(param_list))
        result = eval(op_calc)

        # gradient
        could_gradient = True
        gradient_list = self.param_json.get("gradient", [])
        if len(gradient_list) > 0:
            gradient_calc = "fluid.backward.calc_gradient(result, data_list)"
            exec("{0} = {1}".format(",".join(["g%d" % index if has_gradient else "_"\
                 for index, has_gradient in enumerate(gradient_list)]), gradient_calc))
        else:
            could_gradient = False

        prog = fluid.default_main_program()
        for var in prog.list_vars():
            print var.name

        # execute
        if self.param_json.get("gpu", 0):
            core = fluid.core.CUDAPlace(0)
            exe = fluid.Executor(core)
            exe.run(fluid.default_startup_program())
        else:
            core = fluid.core.CPUPlace()
            exe = fluid.Executor(core)
            exe.run(fluid.default_startup_program())
        feed_list = []
        for i in range(len(feed_key)):
            feed_list.append("'{}':feed_value[{}]".format(feed_key[i], i))

        times = 0
        begin_time = time.time()
        while time.time() - begin_time < 3600:
            times += 1
            if times % 100000 == 0:
                print "times ", times, "costs", time.time() - begin_time
            if type(result) is paddle.fluid.framework.Variable:
                eval("exe.run(feed={%s}, fetch_list=[result.name])" % ",".join(feed_list))
            elif type(result) is tuple or type(result) is list:
                fetch_list = []
                for i in range(len(result)):
                    if type(result[i]) is paddle.fluid.framework.Variable:
                        fetch_list.append("result[{}].name".format(i))
                eval("exe.run(feed={%s}, fetch_list=[%s])"\
                            % (",".join(feed_list), ",".join(fetch_list)))
            else:
                raise(Exception("output error"))
            if times == 1:
                begin_memory = self.get_memory(self.param_json.get("gpu", 0))
        end_memory = self.get_memory(self.param_json.get("gpu", 0))
        result_dict["memory"] = "{}MB".format(end_memory - begin_memory)
        print "times ", times

    def test_op_fast(self):
        """
        op test fast
        """
        export_data = {}
	op_name = self.param_json["op"]
        ret_dict = self.prepare(op_name, self.param_json["params"])
        feed_key = ret_dict["feed_key"]
        feed_value = ret_dict["feed_value"]
        param_list = ret_dict["param_list"]
        data_list = ret_dict["data_list"]
        tf_feed_key = ret_dict["tf_feed_key"]
        tf_param_list = ret_dict["tf_param_list"]
        tf_data_list = ret_dict["tf_data_list"]
        exec_dict = ret_dict["exec_dict"]

        for exec_cmd in exec_dict["feed"]:
            exec(exec_cmd)

        for i in range(len(feed_value)):
            export_data[feed_value[i]] = eval(feed_value[i])
            feed_value[i] = eval(feed_value[i])

        for exec_cmd in exec_dict["paddle"]:
            exec(exec_cmd)

        for i in range(len(data_list)):
            data_list[i] = eval(data_list[i])

        paddle_ret = self.test_op_stablity(feed_key, feed_value, param_list, data_list)
        if "tf-op" in self.param_json and "cover" not in self.param_json:
            self.test_op_function(paddle_ret, tf_feed_key, feed_value,
                                  tf_param_list, tf_data_list, exec_dict["tf"])
        else:
            result_dict["function"] = "-"

        if len(export_data) > 0:
            exec('numpy.savez("{}/{}", {})'.format(sys.argv[4], sys.argv[2],
                 ", ".join(["{}={}".format(d, 'export_data["{}"]'.format(d)) for d in export_data])))

    def test_op_format(self):
        """
        test two formats lead to the same result
        """
	op_name = self.param_json["op"]
        ret_dict = self.prepare(op_name, self.param_json["params"])
        feed_key = ret_dict["feed_key"]
        feed_value = ret_dict["feed_value"]
        param_list = ret_dict["param_list"]
        data_list = ret_dict["data_list"]
        exec_dict = ret_dict["exec_dict"]

        for exec_cmd in exec_dict["feed"]:
            exec(exec_cmd)
        for i in range(len(feed_value)):
            feed_value[i] = eval(feed_value[i])
        for exec_cmd in exec_dict["paddle"]:
            exec(exec_cmd)
        for i in range(len(data_list)):
            data_list[i] = eval(data_list[i])
        ret1 = self.get_calc_ret(feed_key, feed_value, param_list, data_list)
        data_format = "NHWC"
        for i in range(len(exec_dict["format"])):
            if exec_dict["format"][i] == 1:
                if feed_value[i].ndim == 4:
                    #feed_value[i] = feed_value[i].transpose(0, 2, 3, 1)
                    data_list[i] = fluid.layers.transpose(data_list[i], [0, 2, 3, 1])
                    data_format = "NHWC"
                elif feed_value[i].ndim == 5:
                    #feed_value[i] = feed_value[i].transpose(0, 2, 3, 4, 1)
                    data_list[i] = fluid.layers.transpose(data_list[i], [0, 2, 3, 4, 1])
                    data_format = "NDHWC"
                else:
                    raise Exception("code error")
        if "data_layout" in param_list[-1]:
            param_list[-1] = "data_layout='{}'".format(data_format)        
        else:
            param_list.append("data_format='{}'".format(data_format))
        ret2 = self.get_calc_ret(feed_key, feed_value, param_list, data_list)
        self.check_output_equal(len(ret1), len(ret2))
        for i in range(len(ret1)):
            if ret2[i].ndim == 4:
                self.check_output_equal(ret1[i], ret2[i].transpose(0, 3, 1, 2))
            elif ret2[i].ndim == 5:
                self.check_output_equal(ret1[i], ret2[i].transpose(0, 4, 1, 2, 3))
            else:
                raise Exception("code error")

    def get_calc_ret(self, feed_key, feed_value, param_list, data_list):
        """
        calc op
        """
        op_name = self.param_json["op"]
        # define op
        op_calc = "{} = {}({})".format(",".join(["result" if return_flag else "_" for return_flag in self.param_json.get("return", [1])]), 
                                       "fluid.layers." + op_name if "." not in op_name else op_name, ",".join(param_list))
        print op_calc
        exec(op_calc)
        # gradient
        gradient_list = self.param_json.get("gradient", [])
        could_gradient = True
        if len(gradient_list) > 0:
            gradient_calc = "fluid.backward.calc_gradient(result, data_list)"
            cmd = "{0} = {1}".format(",".join(["g%d" % index if has_gradient else "_"\
                  for index, has_gradient in enumerate(gradient_list)]), gradient_calc)
            exec(cmd)
        else:
            could_gradient = False

        prog = fluid.default_main_program()
        for var in prog.list_vars():
            print var.name

        # execute
        if self.param_json.get("gpu", 0):
            core = fluid.core.CUDAPlace(0)
            exe = fluid.Executor(core)
            exe.run(fluid.default_startup_program())
        else:
            core = fluid.core.CPUPlace()
            exe = fluid.Executor(core)
            exe.run(fluid.default_startup_program())
        feed_list = []
        for i in range(len(feed_key)):
            feed_list.append("'{}':feed_value[{}]".format(feed_key[i], i))

        if could_gradient:
            gradient_fetch_list = []
            for index, has_gradient in enumerate(gradient_list):
                if has_gradient:
                    gradient_fetch_list.append("g{}".format(index))
            gradient_fetch = "," + ",".join(gradient_fetch_list)
        else:
            gradient_fetch = ""

        if type(result) is paddle.fluid.framework.Variable:
            cmd = "exe.run(feed={{{}}}, fetch_list=[result.name])".format(
                  ",".join(feed_list), gradient_fetch)
        elif type(result) is tuple or type(result) is list:
            fetch_list = []
            for j in range(len(result)):
                if type(result[j]) is paddle.fluid.framework.Variable:
                    fetch_list.append("result[{}].name".format(j))
            cmd = "exe.run(feed={{{}}}, fetch_list=[{}{}])"\
                  .format(",".join(feed_list), ",".join(fetch_list), gradient_fetch)
        else:
            print "result type:", type(result)
            raise(Exception("output error"))
        print cmd
        return(eval(cmd))


    def test_generate_paddle_performance_code(self):
        """
        generate paddle performance code
        """
        export_codes = {"part0": [], "part1": [], "part2": [], "part3": [], "part4": [], "part5": []}
        export_data = {}
        op_name = self.param_json["op"]
        ret_dict = self.prepare(op_name, self.param_json["params"])
        feed_key = ret_dict["feed_key"]
        feed_value = ret_dict["feed_value"]
        param_list = ret_dict["param_list"]
        data_list = ret_dict["data_list"]
        exec_dict = ret_dict["exec_dict"]

        for exec_cmd in exec_dict["feed"]:
            exec(exec_cmd)
            export_codes["part0"].append('        #' + exec_cmd)

        if len(feed_value) > 0:
            export_codes["part0"].append('        data_load = numpy.load(sys.argv[0].replace(".py", ".npz"))')
        for i in range(len(feed_value)):
            export_data[feed_value[i]] = eval(feed_value[i])
            export_codes["part0"].append('        {0} = data_load["{0}"]'.format(feed_value[i]))

        if len(feed_value) > 0:
            export_codes["part0"].append("        feed_value = [{}]".format(", ".join(feed_value)))

        for exec_cmd in exec_dict["paddle"]:
            export_codes["part0"].append("        " + exec_cmd)

        if len(data_list) > 0:
            export_codes["part0"].append("        data_list = [{}]".format(", ".join(data_list)))

        # define op
        op_calc = "{} = {}({})".format(",".join(["result" if return_flag else "_" for return_flag in \
                  self.param_json.get("return", [1])]), "fluid.layers." + op_name \
                  if "." not in op_name else op_name, ",".join(param_list))
        export_codes["part1"].append("        {}".format(op_calc))
        export_codes["part4"].append("        {}".format(op_calc))
        # gradient
        gradient_list = self.param_json.get("gradient", [])
        could_gradient = True
        if len(gradient_list) > 0:
            gradient_calc = "fluid.backward.calc_gradient(result, data_list)"
            cmd = "{0} = {1}".format(",".join(["g%d" % index if has_gradient else "_"\
                  for index, has_gradient in enumerate(gradient_list)]), gradient_calc)
            export_codes["part1"].append("        " + cmd)
        else:
            could_gradient = False

        export_codes["part2"].append(str(self.param_json.get("gpu", 0)))
        feed_list = []
        for i in range(len(feed_key)):
            feed_list.append("'{}':feed_value[{}]".format(feed_key[i], i))

        if could_gradient:
            gradient_fetch_list = []
            for index, has_gradient in enumerate(gradient_list):
                if has_gradient:
                    gradient_fetch_list.append("g{}".format(index))
            gradient_fetch = "," + ",".join(gradient_fetch_list)
        else:
            gradient_fetch = ""

        return_flag = self.param_json.get("return", [])
        if len(return_flag) == 0:
            cmd = "exe.run(feed={{{}}}, fetch_list=[result.name{}], return_numpy={}, use_program_cache=True)".format(
              ",".join(feed_list), gradient_fetch, "True" if self.param_json.get("return_numpy", 1) else "False")
            export_codes["part3"].append(cmd)
            cmd = "exe.run(feed={{{}}}, fetch_list=[result.name], return_numpy={}, use_program_cache=True)".format(
              ",".join(feed_list), "True" if self.param_json.get("return_numpy", 1) else "False")
            export_codes["part5"].append(cmd)
        else:
            fetch_list = []
            for j in range(len(return_flag)):
                fetch_list.append("result[{}].name".format(j))
            cmd = "exe.run(feed={{{}}}, fetch_list=[{}{}], return_numpy={}, use_program_cache=True)"\
                  .format(",".join(feed_list), ",".join(fetch_list), gradient_fetch, "True" if self.param_json.get("return_numpy", 1) else "False")
            export_codes["part3"].append(cmd)
            cmd = "exe.run(feed={{{}}}, fetch_list=[{}], return_numpy={}, use_program_cache=True)"\
                  .format(",".join(feed_list), ",".join(fetch_list), "True" if self.param_json.get("return_numpy", 1) else "False")
            export_codes["part5"].append(cmd)
        with open("code_template/template_for_paddle_performance", "r") as f:
            template = f.read()
        with open("{}/{}.py".format(sys.argv[4], sys.argv[2]), "w") as f:
            f.write(template.format("\n".join(export_codes["part0"]), "\n".join(export_codes["part1"]),
                                    "\n".join(export_codes["part2"]), "\n".join(export_codes["part3"]),
                                    "\n".join(export_codes["part4"]), "\n".join(export_codes["part5"])))
        if len(export_data) > 0:
            exec('numpy.savez("{}/{}", {})'.format(sys.argv[4], sys.argv[2],
                 ", ".join(["{}={}".format(d, 'export_data["{}"]'.format(d)) for d in export_data])))

    def test_generate_function_code(self):
        """
        generate function code
        """
        export_codes = {"part0": [], "part1": [], "part2": [], "part3": [], "part4": [], "part5": [], "part6":[], "part7":[], "part8":[]}
        export_data = {}
        op_name = self.param_json["op"]
        ret_dict = self.prepare(op_name, self.param_json["params"])
        feed_key = ret_dict["feed_key"]
        feed_value = ret_dict["feed_value"]
        param_list = ret_dict["param_list"]
        data_list = ret_dict["data_list"]
        tf_feed_key = ret_dict["tf_feed_key"]
        tf_param_list = ret_dict["tf_param_list"]
        tf_data_list = ret_dict["tf_data_list"]
        exec_dict = ret_dict["exec_dict"]

        for exec_cmd in exec_dict["feed"]:
            exec(exec_cmd)
            export_codes["part0"].append('        #' + exec_cmd)

        if len(feed_value) > 0:
            export_codes["part0"].append('        data_load = numpy.load(sys.argv[0].replace(".py", ".npz"))')
        for i in range(len(feed_value)):
            export_data[feed_value[i]] = eval(feed_value[i])
            export_codes["part0"].append('        {0} = data_load["{0}"]'.format(feed_value[i]))

        if len(feed_value) > 0:
            export_codes["part0"].append("        feed_value = [{}]".format(", ".join(feed_value)))

        for exec_cmd in exec_dict["paddle"]:
            export_codes["part0"].append("        " + exec_cmd)

        if len(data_list) > 0:
            export_codes["part0"].append("        data_list = [{}]".format(", ".join(data_list)))

        for exec_cmd in exec_dict["tf"]:
            export_codes["part0"].append("        " + exec_cmd)

        if len(tf_data_list) > 0:
            export_codes["part0"].append("        tf_data_list = [{}]".format(", ".join(tf_data_list)))

        # define op
        return_flag_list =  self.param_json.get("return", [])
        if len(return_flag_list) == 0:
            op_calc = "{} = {}({})".format("result0", "fluid.layers." + op_name if "." not in op_name else op_name, ",".join(param_list))
        elif len(return_flag_list) == 1:
            op_calc = "[{}] = {}({})".format("result0", "fluid.layers." + op_name if "." not in op_name else op_name, ",".join(param_list))
        else:
            op_calc = "{} = {}({})".format(",".join(["result%d" % index if return_flag else "_" for index, return_flag in \
                      enumerate(return_flag_list)]), "fluid.layers." + op_name \
                      if "." not in op_name else op_name, ",".join(param_list))
        export_codes["part1"].append("        {}".format(op_calc))
        # gradient
        gradient_list = self.param_json.get("gradient", [])
        could_gradient = True
        if len(gradient_list) > 0:
            gradient_calc = "fluid.backward.calc_gradient(result0, data_list)"
            cmd = "{0} = {1}".format(",".join(["g%d" % index if has_gradient else "_"\
                  for index, has_gradient in enumerate(gradient_list)]), gradient_calc)
            export_codes["part1"].append("        " + cmd)
        else:
            could_gradient = False

        export_codes["part2"].append(str(self.param_json.get("gpu", 0)))
        feed_list = []
        for i in range(len(feed_key)):
            feed_list.append("'{}':feed_value[{}]".format(feed_key[i], i))

        if could_gradient:
            gradient_fetch_list = []
            for index, has_gradient in enumerate(gradient_list):
                if has_gradient:
                    gradient_fetch_list.append("g{}".format(index))
            gradient_fetch = "," + ",".join(gradient_fetch_list)
        else:
            gradient_fetch = ""

        if len(return_flag_list) == 1:
            cmd = "exe.run(feed={{{}}}, fetch_list=[result0.name{}], return_numpy={}, use_program_cache=True)".format(
              ",".join(feed_list), gradient_fetch, "True" if self.param_json.get("return_numpy", 1) else "False")
            export_codes["part3"].append(cmd)
        else:
            fetch_list = []
            for index, return_flag in enumerate(return_flag_list):
                if return_flag:
                    fetch_list.append("result{}".format(index))
            cmd = "exe.run(feed={{{}}}, fetch_list=[{}{}], return_numpy={}, use_program_cache=True)"\
                  .format(",".join(feed_list), ",".join(fetch_list), gradient_fetch, "True" if self.param_json.get("return_numpy", 1) else "False")
            export_codes["part3"].append(cmd)

        if "tf-op" in self.param_json and self.param_json.get("tf-disable", 0) == 0:
            # define tf-op
            op_calc = "{}({})".format(self.param_json["tf-op"], ",".join(tf_param_list))
            export_codes["part4"].append("        tf_result = {}".format(op_calc))
            gradient_list = self.param_json.get("gradient", [])
            gradient_fetch_list = [] 
            if len(gradient_list) > 0: 
                gradient_calc = "tf.gradients(tf_result, tf_data_list)"
                export_codes["part4"].append("        tf_gradient = {}".format(gradient_calc))
                for index, tf_g in enumerate(gradient_list):
                    if tf_g:
                        gradient_fetch_list.append("tf_gradient[{}]".format(index))
            tf_feed_list = []
            for i in range(len(tf_feed_key)):
                tf_feed_list.append("{}:feed_value[{}]".format(tf_feed_key[i], i))

            export_codes["part5"].append("{'GPU': %d}" % self.param_json.get("gpu", 0))
            export_codes["part6"].append("        tf_ret = sess.run([tf_result, {}], feed_dict={{{}}})".format(\
                                         ",".join(gradient_fetch_list), ",".join(tf_feed_list)))
            export_codes["part7"].append(str(sum(gradient_list)))
            export_codes["part8"].append(str(self.param_json.get("level-diff", 0)))

            with open("code_template/template_for_function", "r") as f:
                template = f.read()
            with open("{}/{}.py".format(sys.argv[4], sys.argv[2]), "w") as f:
                f.write(template.format("\n".join(export_codes["part0"]), "\n".join(export_codes["part1"]),
                                        "\n".join(export_codes["part2"]), "\n".join(export_codes["part3"]),
                                        "\n".join(export_codes["part4"]), "\n".join(export_codes["part5"]),
                                        "\n".join(export_codes["part6"]), "\n".join(export_codes["part7"]),
                                        "\n".join(export_codes["part8"])))
        else:
            with open("code_template/template_for_consistency", "r") as f:
                template = f.read()
            with open("{}/{}.py".format(sys.argv[4], sys.argv[2]), "w") as f:
                f.write(template.format("\n".join(export_codes["part0"]), "\n".join(export_codes["part1"]),
                                        "\n".join(export_codes["part2"]), "\n".join(export_codes["part3"])))
        if len(export_data) > 0:
            exec('numpy.savez("{}/{}", {})'.format(sys.argv[4], sys.argv[2],
                 ", ".join(["{}={}".format(d, 'export_data["{}"]'.format(d)) for d in export_data])))

    def test_op_performance(self):
        """
        op performance test
        """
        op_name = self.param_json["op"]
        ret_dict = self.prepare(op_name, self.param_json["params"])
        feed_key = ret_dict["feed_key"]
        feed_value = ret_dict["feed_value"]
        param_list = ret_dict["param_list"]
        data_list = ret_dict["data_list"]
        tf_feed_key = ret_dict["tf_feed_key"]
        tf_param_list = ret_dict["tf_param_list"]
        tf_data_list = ret_dict["tf_data_list"]
        exec_dict = ret_dict["exec_dict"]

        for exec_cmd in exec_dict["feed"]:
            exec(exec_cmd)

        for i in range(len(feed_value)):
            feed_value[i] = eval(feed_value[i])

        for exec_cmd in exec_dict["paddle"]:
            exec(exec_cmd)

        for i in range(len(data_list)):
            data_list[i] = eval(data_list[i])
        
        # define op
        op_calc = "{} = {}({})".format(",".join(["result" if return_flag else "_" for return_flag \
                  in self.param_json.get("return", [1])]), "fluid.layers." + op_name \
                  if "." not in op_name else op_name, ",".join(param_list))
        exec(op_calc)
        # gradient
        gradient_list = self.param_json.get("gradient", [])
        could_gradient = True
        if len(gradient_list) > 0:
            gradient_calc = "fluid.backward.calc_gradient(result, data_list)"
            cmd = "{0} = {1}".format(",".join(["g%d" % index if has_gradient else "_"\
                  for index, has_gradient in enumerate(gradient_list)]), gradient_calc)
            exec(cmd)
        else:
            could_gradient = False

        # execute
        if self.param_json.get("gpu", 0):
            core = fluid.core.CUDAPlace(0)
            exe = fluid.Executor(core)
            exe.run(fluid.default_startup_program())
        else:
            core = fluid.core.CPUPlace()
            exe = fluid.Executor(core)
            exe.run(fluid.default_startup_program())
        feed_list = []
        for i in range(len(feed_key)):
            feed_list.append("'{}':feed_value[{}]".format(feed_key[i], i))

        outs = []
        performance = []

        if could_gradient:
            gradient_fetch_list = []
            for index, has_gradient in enumerate(gradient_list):
                if has_gradient:
                    gradient_fetch_list.append("g{}".format(index))
            gradient_fetch = "," + ",".join(gradient_fetch_list)
        else:
            gradient_fetch = ""

        if type(result) is paddle.fluid.framework.Variable:
            cmd = "exe.run(feed={{{}}}, fetch_list=[result.name{}], return_numpy={}, use_program_cache=True)".format(
                  ",".join(feed_list), gradient_fetch, "True" if self.param_json.get("return_numpy", 1) else "False")
        elif type(result) is tuple or type(result) is list:
            fetch_list = []
            for j in range(len(result)):
                if type(result[j]) is paddle.fluid.framework.Variable:
                    fetch_list.append("result[{}].name".format(j))
            cmd = "exe.run(feed={{{}}}, fetch_list=[{}{}], return_numpy={}, use_program_cache=True)"\
                  .format(",".join(feed_list), ",".join(fetch_list), gradient_fetch, "True" if self.param_json.get("return_numpy", 1) else "False")
        else:
            print "result type:", type(result)
            raise(Exception("output error"))
        
        for i in range(50):
            before_time = time.time()
            outs.append(eval(cmd))
            after_time = time.time()
            performance.append(after_time - before_time)
        result_dict["performance"] = "paddle=%.06f" % (sum(performance[1:]) / (len(performance) - 1))
        print performance
        print result_dict["performance"]
        

    def test_op_stablity(self, feed_key, feed_value, param_list, data_list):
        """
        op stablity test
        """
        op_name = self.param_json["op"]
        # define op
        op_calc = "result = {}({})".format("fluid.layers." + op_name \
                  if "." not in op_name else op_name, ",".join(param_list))
        print op_calc
        exec(op_calc)
        # gradient
        gradient_list = self.param_json.get("gradient", [])
        could_gradient = True
        if len(gradient_list) > 0:
            gradient_calc = "fluid.backward.calc_gradient(result, data_list)"
            cmd = "{0} = {1}".format(",".join(["g%d" % index if has_gradient else "_"\
                  for index, has_gradient in enumerate(gradient_list)]), gradient_calc)
            exec(cmd)
        else:
            could_gradient = False

        prog = fluid.default_main_program()
        for var in prog.list_vars():
            print var.name

        # execute
        if self.param_json.get("gpu", 0):
            core = fluid.core.CUDAPlace(0)
            exe = fluid.Executor(core)
            exe.run(fluid.default_startup_program())
            # cpu core to compare with gpu core
            cpu_core = fluid.core.CPUPlace()
            cpu_exe = fluid.Executor(cpu_core)
            cpu_exe.run(fluid.default_startup_program())
        else:
            core = fluid.core.CPUPlace()
            exe = fluid.Executor(core)
            exe.run(fluid.default_startup_program())
        feed_list = []
        for i in range(len(feed_key)):
            feed_list.append("'{}':feed_value[{}]".format(feed_key[i], i))

        outs = []
        if could_gradient:
            gradient_fetch_list = []
            for index, has_gradient in enumerate(gradient_list):
                if has_gradient:
                    gradient_fetch_list.append("g{}".format(index))
            gradient_fetch = "," + ",".join(gradient_fetch_list)
        else:
            gradient_fetch = ""

        if type(result) is paddle.fluid.framework.Variable:
            cmd = "exe.run(feed={{{}}}, fetch_list=[result.name{}], return_numpy={}, use_program_cache=True)".format(
                  ",".join(feed_list), gradient_fetch, "True" if self.param_json.get("return_numpy", 1) else "False")
        elif type(result) is tuple or type(result) is list:
            fetch_list = []
            for j in range(len(result)):
                if type(result[j]) is paddle.fluid.framework.Variable:
                    fetch_list.append("result[{}].name".format(j))
            cmd = "exe.run(feed={{{}}}, fetch_list=[{}{}], return_numpy={}, use_program_cache=True)"\
                  .format(",".join(feed_list), ",".join(fetch_list), gradient_fetch, "True" if self.param_json.get("return_numpy", 1) else "False")
        else:
            print "result type:", type(result)
            raise(Exception("output error"))
        
        for i in range(10):
            if i == 1 and self.param_json.get("gpu", 0):
                outs.append(eval("cpu_" + cmd))
            else:
                outs.append(eval(cmd))
        
        # assert result
        stable = True
        for i in range(1, 10):
            if not stable:
                break
            print "ASSERT EQUAL {} and {}".format(0, i)
            for j in range(len(outs[0])):
                try:
                    print "ASSERT {}th ret".format(j)
                    err_msg = "cpu and gpu diff" if i == 1 else "multi times consistency diff"
                    numpy.testing.assert_allclose(outs[0][j], outs[i][j], atol=0, rtol=1e-6, err_msg=err_msg)
                    #self.check_output_equal(outs[0][j], outs[i][j], places=self.param_json.get("places", 6))
                except (AssertionError, ValueError, TypeError) as e:
                    traceback.print_exc()
                    stable = False
                    break
        if stable:
            result_dict["stablity"] = "PASS"
        return outs[0]

    def test_generate_tf_performance_code(self):
        """
        generate tf performance code
        """
        if "tf-op" not in self.param_json or self.param_json.get("tf-disable", 0):
        # no tf or disable tf, just return
            return
        export_data = {}
        export_codes = {"part0": [], "part1": [], "part2": [], "part3": [], "part4": [], "part5": []}
        op_name = self.param_json["op"]
        ret_dict = self.prepare(op_name, self.param_json["params"])
        feed_value = ret_dict["feed_value"]
        tf_feed_key = ret_dict["tf_feed_key"]
        tf_param_list = ret_dict["tf_param_list"]
        tf_data_list = ret_dict["tf_data_list"]
        exec_dict = ret_dict["exec_dict"]

        for exec_cmd in exec_dict["feed"]:
            exec(exec_cmd)
            export_codes["part0"].append('        #' + exec_cmd)

        if len(feed_value) > 0:
            export_codes["part0"].append('        data_load = numpy.load(sys.argv[0].replace(".py", ".npz"))')
        for i in range(len(feed_value)):
            export_data[feed_value[i]] = eval(feed_value[i])
            export_codes["part0"].append('        {0} = data_load["{0}"]'.format(feed_value[i]))

        if len(feed_value) > 0:
            export_codes["part0"].append("        feed_value = [{}]".format(", ".join(feed_value)))

        for exec_cmd in exec_dict["tf"]:
            export_codes["part0"].append("        " + exec_cmd)

        if len(tf_data_list) > 0:
            export_codes["part0"].append("        tf_data_list = [{}]".format(", ".join(tf_data_list)))

        # define tf-op
        op_calc = "{}({})".format(self.param_json["tf-op"], ",".join(tf_param_list))
        export_codes["part1"].append("        tf_result = {}".format(op_calc))
        export_codes["part4"].append("        tf_result = {}".format(op_calc))
        gradient_list = self.param_json.get("gradient", [])
        gradient_fetch_list = []
        if len(gradient_list) > 0:
            gradient_calc = "tf.gradients(tf_result, tf_data_list)"
            export_codes["part1"].append("        tf_gradient = {}".format(gradient_calc))
            for index, tf_g in enumerate(gradient_list):
                if tf_g:
                    gradient_fetch_list.append("tf_gradient[{}]".format(index))
        tf_feed_list = []
        for i in range(len(tf_feed_key)):
            tf_feed_list.append("{}:feed_value[{}]".format(tf_feed_key[i], i))

        export_codes["part2"].append("{'GPU': %d}" % self.param_json.get("gpu", 0))
        export_codes["part3"].append("sess.run([tf_result, {}], feed_dict={{{}}})".format(\
                                     ",".join(gradient_fetch_list), ",".join(tf_feed_list)))
        export_codes["part5"].append("sess.run([tf_result], feed_dict={{{}}})".format(",".join(tf_feed_list)))
        with open("code_template/template_for_tf_performance", "r") as f:
            template = f.read()
        with open("{}/{}.py".format(sys.argv[4], sys.argv[2]), "w") as f:
            f.write(template.format("\n".join(export_codes["part0"]), "\n".join(export_codes["part1"]),
                                    "\n".join(export_codes["part2"]), "\n".join(export_codes["part3"]),
                                    "\n".join(export_codes["part4"]), "\n".join(export_codes["part5"])))
        if len(export_data) > 0:
            exec('numpy.savez("{}/{}", {})'.format(sys.argv[4], sys.argv[2],
                 ", ".join(["{}={}".format(d, 'export_data["{}"]'.format(d)) for d in export_data])))

    def test_op_function(self, paddle_ret, tf_feed_key, feed_value, tf_param_list, tf_data_list, exec_list):
        """
        op function test
        """
        op_name = self.param_json["op"]
        for exec_cmd in exec_list:
            exec(exec_cmd)
        for i in range(len(tf_data_list)):
            tf_data_list[i] = eval(tf_data_list[i])

        gradient_list = self.param_json.get("gradient", [])
        could_gradient = len(gradient_list) > 0
        tf_could_gradient = True
        front = True
        back = True

        # define tf-op
        op_calc = "{}({})".format(self.param_json["tf-op"], ",".join(tf_param_list))
        tf_result = eval(op_calc)
        if len(gradient_list) > 0:
            gradient_calc = "tf.gradients(tf_result, tf_data_list)"
            tf_gradient = eval(gradient_calc)
            print "tf gradient return:", tf_gradient
            l = 0
            while l < len(tf_gradient):
                if tf_gradient[l] is not None:
                    break
                l += 1
            if l == len(tf_gradient):
                tf_could_gradient = False
        else:
            tf_could_gradient = False

        device_count = self.param_json.get("gpu", 0)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, device_count={'GPU': device_count}))
        tf_feed_list = []
        for i in range(len(tf_feed_key)):
            tf_feed_list.append("{}:feed_value[{}]".format(tf_feed_key[i], i))
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        # execute
        before_time = time.time()
        gradient_fetch_list = []
        if could_gradient and tf_could_gradient:
            for index, tf_g in enumerate(gradient_list):
                if tf_g:
                    gradient_fetch_list.append("tf_gradient[{}]".format(index))
        exec("tf_ret = sess.run([tf_result, {}], feed_dict={{{}}})".format(",".join(gradient_fetch_list), ",".join(tf_feed_list)))
        after_time = time.time()
        result_dict["performance"] += "\ntf=%.06f" % (after_time - before_time)

        diff_level = self.param_json.get("level-diff", 0)
        gradient_count = sum(gradient_list)
        if gradient_count > 0:
            paddle_result = paddle_ret[:gradient_count * -1]
            tf_result = tf_ret[:gradient_count * -1]
        else:
            paddle_result = paddle_ret
            tf_result = tf_ret
        while diff_level > 0:
            paddle_result = paddle_result[0]
            diff_level -= 1
    
        while diff_level < 0:
            tf_result = tf_result[0]
            diff_level += 1

        if self.param_json.get("debug", 0):
            print "tf:",
            pprint.pprint(tf_result)
            print "paddle:",
            pprint.pprint(paddle_result)
        try:
            numpy.testing.assert_allclose(paddle_result, tf_result, atol=0, rtol=1e-6, err_msg="compared with tf: forward diff")
            #self.check_output_equal(paddle_result, tf_result, places=self.param_json.get("places", 6))
        except (AssertionError, ValueError, TypeError) as e:
            front = False
            traceback.print_exc()
        
        if could_gradient and tf_could_gradient:
            if self.param_json.get("debug", 0):
                print "paddle gradient:",
                print paddle_ret[gradient_count * -1:]
                print "tf gradient:",
                print tf_ret[gradient_count * -1:]
            try:
                for i in range(gradient_count):
                    numpy.testing.assert_allclose(paddle_ret[-1 - i], tf_ret[-1 - i], atol=0, rtol=1e-6, err_msg="compared with tf: backward diff")
            except (AssertionError, ValueError, TypeError) as e:
                back = False
                traceback.print_exc()
        
        if could_gradient and tf_could_gradient:
            if front and back:
                result_dict["function"] = "PASS(Front&Back)"
            elif not front and back:
                result_dict["function"] = "FAIL(Front)"
            elif front and not back:
                result_dict["function"] = "FAIL(Back)"
            else:
                result_dict["function"] = "FAIL(Front&Back)"
        else:
            if front:
                result_dict["function"] = "PASS(Front)"
            else:
                result_dict["function"] = "FAIL(Front)"

    def test_op_dygraph(self):
        """
        op dygraph test
        :to-do:
        """
        op_name = self.param_json["op"]
        ret_dict = self.prepare(op_name, self.param_json["params"], test_type="dygraph")
        feed_key = ret_dict["feed_key"]
        feed_value = ret_dict["feed_value"]
        param_list = ret_dict["param_list"]
        data_list = ret_dict["data_list"]
        exec_dict = ret_dict["exec_dict"]

        for exec_cmd in exec_dict["feed"]:
            exec(exec_cmd)

        for i in range(len(feed_value)):
            feed_value[i] = eval(feed_value[i])

        for exec_cmd in exec_dict["paddle"]:
            exec(exec_cmd)

        for i in range(len(data_list)):
            data_list[i] = eval(data_list[i])

        if self.param_json.get("gpu", 0):
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()
        with fluid.dygraph.guard(place=place):
            eval_str = "fluid.layers.{}({})".format(op_name, ",".join(param_list))
            print eval_str
            _input = fluid.dygraph.to_variable(feed_value[0])
            x = eval(eval_str)
            dy_out = x.numpy()
	    x.backward()
	    dy_grad = _input.gradient()
        ret = self.get_calc_ret(feed_key, feed_value, param_list, data_list)
        self.check_output_equal(ret, dy_out)

    def check_output_equal(self, output1, output2, places=6):
        """
        compare output1 and output2
        """
        if type(output1) is numpy.ndarray:
            self.check_output_equal(output1.tolist(), output2, places=places)
        elif type(output2) is numpy.ndarray:
            self.check_output_equal(output1, output2.tolist(), places=places)
        elif type(output1) is list and type(output2) is list:
            if len(output1) == len(output2):
                for i in range(len(output1)):
                    self.check_output_equal(output1[i], output2[i], places=places)
            else:
                if len(output1) == 1:
                    self.check_output_equal(output1[0], output2, places=places)
                elif len(output2) == 1:
                    self.check_output_equal(output1, output2[0], places=places)
                else:
                    self.assertTrue(False)
        elif type(output1) is list and len(output1) == 1:
            self.check_output_equal(output1[0], output2, places=places)
        elif type(output2) is list and len(output2) == 1:
            self.check_output_equal(output1, output2[0], places=places)
        elif type(output1) is bool or type(output1) is numpy.bool_:
            self.assertEqual(output1, output2)
        elif type(output1) is float and numpy.isnan(output1):
            self.assertTrue(math.isnan(output2))
        elif type(output1) is float and numpy.isinf(output1):
            self.assertTrue(math.isinf(output2))
        else:
            self.assertTrue(numpy.allclose(output1, output2, rtol=1.e-6, atol=0), msg="{} {}".format(output1, output2))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(ParametrizedTestCase.parametrize(PaddleTest, sys.argv[1], sys.argv[3]))
    runner = unittest.TextTestRunner()
    if os.path.exists("{}/{}.result".format(sys.argv[4], sys.argv[2])):
        with open("{}/{}.result".format(sys.argv[4], sys.argv[2]), "r") as f:
            result_dict = json.load(f)
        if sys.argv[3] == "fast":
            result_dict["function"] = "FAIL"
            result_dict["stablity"] = "FAIL"
        else:
            result_dict["memory"] = "FAIL"
    else:
        result_dict = {
            "function": "FAIL",
            "performance": "-",
            "performance_forward": "-",
            "stablity": "FAIL",
            "memory": "-"
        }
    result = runner.run(suite)
    with open("{}/{}.result".format(sys.argv[4], sys.argv[2]), "w") as f:
        json.dump(result_dict, f, indent=4)

    if len(result.failures) + len(result.errors) > 0:
        sys.exit(1)
