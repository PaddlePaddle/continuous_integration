# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import time
import logging

import paddle.fluid.inference as paddle_infer

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class FullChainLog(object):
    def __init__(self,
                 config,
                 model_info: dict,
                 data_info: dict,
                 perf_info: dict,
                 mem_info: dict,
                 **kwargs):
        """
        __init__
        """
        # FullChainLog Version
        self.log_version = 1.0

        # model info
        self.model_info = model_info
        self.model_name = model_info['model_name']
        self.precision = model_info['precision']

        # data info
        self.data_info = data_info
        self.batch_size = data_info['batch_size']
        self.shape = data_info['shape']
        self.data_num = data_info['data_num']

        # conf info
        self.config_status = self.parse_config(config)

        # perf info
        self.perf_info = perf_info
        self.preprocess_time = round(perf_info['preprocess_time'], 4)
        self.inference_time = round(perf_info['inference_time'], 4)
        self.postprocess_time = round(perf_info['postprocess_time'], 4)
        self.total_time = round(perf_info['total_time'], 4)

        # mem info
        self.mem_info = mem_info
        self.cpu_rss = mem_info['cpu_rss']
        self.gpu_rss = mem_info['gpu_rss']
        self.gpu_util = mem_info['gpu_util']

    def parse_config(self, config) -> dict:
        """
        parse paddle predictor config
        """
        config_status = {}
        config_status['runtime_device'] = "gpu" if config.use_gpu() else "cpu"
        config_status['ir_optim'] = config.ir_optim()
        config_status['enable_tensorrt'] = config.tensorrt_engine_enabled()
        config_status['precision'] = self.precision
        config_status['enable_mkldnn'] = config.mkldnn_enabled()
        config_status[
            'cpu_math_library_num_threads'] = config.cpu_math_library_num_threads(
            )
        return config_status

    def report(self):
        """
        print log report
        """
        logger.info("\n")
        logger.info("---------------------- Paddle info ----------------------")
        logger.info(f"{paddle_infer.get_version()}")
        logger.info(f"log_style_version: {self.log_version}")
        logger.info("----------------------- Conf info -----------------------")
        logger.info(f"runtime_device: {self.config_status['runtime_device']}")
        logger.info(f"ir_optim: {self.config_status['ir_optim']}")
        logger.info(f"enable_memory_optim: {True}")
        logger.info(f"enable_tensorrt: {self.config_status['enable_tensorrt']}")
        logger.info(f"precision: {self.precision}")
        logger.info(f"enable_mkldnn: {self.config_status['enable_mkldnn']}")
        logger.info(
            f"cpu_math_library_num_threads: {self.config_status['cpu_math_library_num_threads']}"
        )
        logger.info("----------------------- Model info ----------------------")
        logger.info(f"model_name: {self.model_name}")
        logger.info("----------------------- Data info -----------------------")
        logger.info(f"batch_size: {self.batch_size}")
        logger.info(f"input_shape: {self.shape}")
        logger.info(f"data_num: {self.data_num}")
        logger.info("----------------------- Perf info -----------------------")
        logger.info(
            f"cpu_rss(MB): {int(self.mem_info['cpu_rss'])}, gpu_rss(MB): {int(self.mem_info['gpu_rss'])}, gpu_util: {round(self.mem_info['gpu_util'], 1)}%"
        )
        logger.info(f"total time spent(s): {self.total_time}")
        logger.info(
            f"preproce_time(ms): {round(self.preprocess_time*1000, 1)}, inference_time(ms): {round(self.inference_time*1000, 1)}, postprocess_time(ms): {round(self.postprocess_time*1000, 1)}"
        )

    def print_help(self):
        """
        print function help
        """
        logger.info("""Usage: 
            Print fullchain test logs.
            FullChainLog(config, model_name, batch_size, shape, precision, times, mem_info)
            """)


def main():
    """
    main, sample usage
    """
    model_path = "./"
    config = paddle_infer.Config(model_path + "__model__",
                                 model_path + "_params__")

    model_info = {'model_name': 'AlexNet', 'precision': 'fp32'}
    data_info = {'batch_size': 1, 'shape': '1,3,224,224', 'data_num': 1000}
    perf_info = {
        'preprocess_time': 3.14,
        'inference_time': 2.1,
        'postprocess_time': 5.4,
        'total_time': 7.9
    }
    mem_info = {'cpu_rss': 1.02, 'gpu_rss': 13, 'gpu_util': 80}

    FullChainLog(config, model_info, data_info, perf_info, mem_info).report()


if __name__ == "__main__":
    main()
