# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import os
import sys
import numpy as np
import yaml
import paddle.fluid as fluid

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from helper import Record, JsonInfo
import logging

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class DeployConfig(object):
    """deploy config
    """

    def __init__(self,
                 model_path,
                 batch_size=1,
                 min_subgraph_size=3,
                 trt_dynamic_shape_info=None):
        """
        Args:
            model_path (string): [the path of test model]
            batch_size (int, optional): [description]. Defaults to 1.
            min_subgraph_size (int, optional): [description]. Defaults to 3.
            trt_dynamic_shape_info (namedtuple, optional): [description]. Defaults to None.
        Usages:
            import collections
            min_input_shape = {"image":[3, 32, 156]}
            max_input_shape = {"image":[3, 32, 448]}
            opt_input_shape = {"image":[3, 32, 320]}
            trt_dynamic_shape_info = collections.namedtuple('new_tuple',
                        ['min_input_shape', 'max_input_shape', 'opt_input_shape'])
        Raises:
            Exception: [description]
        """
        if not os.path.exists(model_path):
            raise Exception('Config file path [%s] invalid!' % model_path)

        # 1. get paddle model and params file path
        if os.path.exists(os.path.join(model_path, "__params__")):
            self.model_file = os.path.join(model_path, "__model__")
            self.param_file = os.path.join(model_path, "__params__")
            self.combined_model = True
        else:
            self.combined_model = False
            self.param_file = model_path

        # 2. set trt default config
        self.batch_size = batch_size
        self.min_subgraph_size = min_subgraph_size
        self.trt_dynamic_shape_info = trt_dynamic_shape_info

    def analysis_config(self, config_type):
        """
        init analysis predictor configuration
        Args:
            config_type(str): describe prediction configuration
        Returns:
            predictor_config(AnalysisConfig) : configuration pointer 
        """
        # following configs are only allowed to be used in analysis mode
        trt_precision_map = {
            'trt_fp32': fluid.core.AnalysisConfig.Precision.Float32,
            'trt_fp16': fluid.core.AnalysisConfig.Precision.Half,
            'trt_int8': fluid.core.AnalysisConfig.Precision.Int8
        }

        if self.combined_model:
            predictor_config = fluid.core.AnalysisConfig(self.model_file,
                                                         self.param_file)
        else:
            predictor_config = fluid.core.AnalysisConfig(self.param_file)

        if config_type == 'cpu':
            predictor_config.disable_gpu()
        elif config_type == 'cpu_no_ir':
            predictor_config.disable_gpu()
            predictor_config.switch_ir_optim(False)
            predictor_config.disable_glog_info()
        elif config_type == 'mkldnn':
            predictor_config.disable_gpu()
            predictor_config.enable_mkldnn()
            predictor_config.set_cpu_math_library_num_threads(4)
        elif config_type == 'gpu':
            predictor_config.enable_use_gpu(100, 0)
        elif config_type == 'gpu_no_ir':
            predictor_config.enable_use_gpu(100, 0)
            predictor_config.switch_ir_optim(False)
            predictor_config.disable_glog_info()
        elif config_type == 'lite':
            predictor_config.enable_lite_engine()
        elif config_type in trt_precision_map.keys():
            predictor_config.enable_use_gpu(100, 0)
            predictor_config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=self.batch_size,
                min_subgraph_size=self.min_subgraph_size,
                precision_mode=trt_precision_map[config_type],
                use_static=False,
                use_calib_mode=True if config_type == 'trt_int8' else False)
            if self.trt_dynamic_shape_info is not None:
                # set dynamic shape info
                min_input_shape, max_input_shape, opt_input_shape = self.trt_dynamic_shape_info
                predictor_config.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape)
        else:
            raise Exception('Config type [%s] invalid!' % config_type)
        # predictor_config.switch_ir_optim(True) # default is enabled, no need to enable
        predictor_config.switch_specify_input_names(True)
        predictor_config.enable_memory_optim()  # enable temporarily
        predictor_config.switch_use_feed_fetch_ops(False)
        return predictor_config

    def native_config(self, config_type):
        """
        init native predictor configuration
        Args:
            config_type(string) : cpu, gpu, mkldnn, etc.
        Returns:
            predictor_config(NativeConfig) : configuration
        """
        predictor_config = fluid.core.NativeConfig()
        predictor_config.prog_file = self.model_file
        predictor_config.param_file = self.param_file
        if config_type == 'cpu':
            predictor_config.use_gpu = False
        elif config_type == 'gpu':
            predictor_config.use_gpu = True
        else:
            raise Exception('Config type [%s] invalid for native config!' %
                            config_type)
        predictor_config.device = 0
        predictor_config.fraction_of_gpu_memory = 0
        return predictor_config

    def summary_config(self, predictor_config):
        """
        summary analysis config
        TODO
        Returns:
            [type]: [description]
        """
        if type(predictor_config) is not fluid.core.AnalysisConfig:
            raise Exception('Config [%s] is not fluid.core.AnalysisConfig' %
                            predictor_config)
        else:
            summary_info = {}
            summary_info[
                'cpu_math_library_num_threads'] = predictor_config.cpu_math_library_num_threads(
                )
            summary_info[
                'fraction_of_gpu_memory_for_pool'] = predictor_config.fraction_of_gpu_memory_for_pool(
                )
            summary_info['gpu_device_id'] = predictor_config.gpu_device_id()
            summary_info[
                'lite_engine_enabled'] = predictor_config.lite_engine_enabled()
            summary_info['mkldnn_enabled'] = predictor_config.mkldnn_enabled()
            summary_info[
                'tensorrt_engine_enabled'] = predictor_config.tensorrt_engine_enabled(
                )
            summary_info[
                'use_feed_fetch_ops_enabled'] = predictor_config.use_feed_fetch_ops_enabled(
                )
            summary_info[
                'model_from_memory'] = predictor_config.model_from_memory()
            summary_info['use_gpu'] = predictor_config.use_gpu()
        logger.debug(
            '=================== Analysis config Summary ===================')
        for k, v in summary_info.items():
            logger.debug("predictor_config['{0}'] = {1}".format(k, v))
        logger.debug(
            '===============================================================')


class Predictor(object):
    """init predictor
    """

    def __init__(self,
                 model_path,
                 predictor_mode="Analysis",
                 config_type="cpu",
                 batch_size=1,
                 min_subgraph_size=1,
                 trt_dynamic_shape_info=None):
        """
        init configuration of predictor
        Args:
            model_path(string): the path of test model
            predictor_mode(strings): create native or analysis predictor
            config_type(strings): describe analysis prediction configuration
        """
        configs = DeployConfig(
            model_path=model_path,
            batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            trt_dynamic_shape_info=trt_dynamic_shape_info)
        analysis_predictor_config = configs.analysis_config(config_type)

        logger.debug("analysis_predictor_config : {}".format(
            analysis_predictor_config))
        configs.summary_config(analysis_predictor_config)  # summary configs

        if predictor_mode == "Analysis":
            logger.info("current config is Analysis config")
            predictor0 = fluid.core.create_paddle_predictor(
                analysis_predictor_config)
            # clone main predictor to test predictor.clone api
            self.predictor = predictor0.clone(None)
            logger.info("analysis predictor create and clone successful")
        elif predictor_mode == "Native":
            native_predictor_config = DeployConfig(model_path).native_config(
                config_type)
            logger.info(native_predictor_config)
            logger.info("current config is Native config")
            # use analysis predictor to retrive number of inputs
            analysis_predictor_config.disable_glog_info()
            self.analysis_predictor = fluid.core.create_paddle_predictor(
                analysis_predictor_config)
            # use native predictor to predict
            self.native_predictor = fluid.core.create_paddle_predictor(
                native_predictor_config)
            logger.info("native predictor create successful")

    def analysis_predict(self, json_dir, repeats=1):
        """
        use zero copy and analysis config to predict
        Args:
            json_dir(string) : "*.json"
            repeats(int)
        Returns:
            outputs(list|[numpy.array, numpy.array]): list of numpy array
            ave_time(float): infer speed
        """
        # parse json from data file
        input_info = JsonInfo().parse_json(json_dir)
        # assign data to Tensor
        input_names = self.predictor.get_input_names()
        for i, input_data_name in enumerate(input_names):
            record = Record().load_data_from_json(input_info[i])
            record = next(record)
            logger.info("====> input_names[{0}] = {1} <====".format(
                i, input_names[i]))
            input_tensor = self.predictor.get_input_tensor(input_data_name)
            logger.debug("record.data shape is {}".format(record.data.shape))
            input_tensor.copy_from_cpu(record.data)
            if hasattr(record, 'lod'):
                input_tensor.set_lod([record.lod])

        cost_time = []
        for i in range(repeats):
            t1 = time.time()

            self.predictor.zero_copy_run()
            # get outputs
            outputs = []
            output_names = self.predictor.get_output_names()
            for i, output_data in enumerate(output_names):
                logger.debug("====> output_names[{0}] = {1} <====".format(
                    i, output_data))
                output_tensor = self.predictor.get_output_tensor(output_data)
                _output = output_tensor.copy_to_cpu()
                outputs.append(_output)  # return list[numpy.ndarray]
            # finish whole process of prediction
            t2 = time.time()
            cost_time.append(t2 - t1)
        cost_time = np.array(cost_time)
        ave_time = cost_time.mean()

        logger.info("prediction finished")
        return outputs, ave_time

    def native_predict(self, json_dir, repeats=1):
        """
        use PaddleTensor and native config to predict
        Args:
            json_dir(string) : "*.json"
            repeats(int)
        Returns:
            outputs(list|numpy.array): numpy array
            ave_time(float): infer speed
        """
        # parse json from data file
        input_info = JsonInfo().parse_json(json_dir)
        # assign data to Tensor
        input_names = self.analysis_predictor.get_input_names()
        Tensors = []
        for i, input_data_name in enumerate(input_names):
            record = Record().load_data_from_json(input_info[i])
            record = next(record)
            logger.info("====> input_names[{0}] = {1} <====".format(
                i, input_data_name))
            input_tensor = fluid.core.PaddleTensor(
                name=input_data_name, data=record.data)
            if hasattr(record, 'lod'):
                input_tensor.set_lod(record.lod)
            Tensors.append(input_tensor)

        cost_time = []
        for i in range(repeats):
            t1 = time.time()
            outputs = self.native_predictor.run(Tensors)
            t2 = time.time()
            cost_time.append(t2 - t1)
        cost_time = np.array(cost_time)
        ave_time = cost_time.mean()
        output_data = [x.as_ndarray()
                       for x in outputs]  # return list[numpy.ndarray]
        return output_data, ave_time
