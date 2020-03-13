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
import numpy as np
import paddle
import paddle.fluid as fluid


class InferApiTest(object):
    """
    Basic class of inference API test

    Attributes:
        name: string that indicate model's name
    """

    def __init__(self, name):
        """
        init
        Args:
            name(string) : name of model
        """
        self.name = name
        pass

    def set_config(self, options):
        """
        set_config

        Args:
            options(dict): it contains configurations

        example:
            {"use_gpu": True, "use_tensorrt": False}
        """
        model_name = self.name
        use_gpu = options.get("use_gpu", False)
        use_tensorrt = options.get("use_tensorrt", False)
        use_anakin = options.get("use_anakin", False)
        gpu_memory = options.get("gpu_memory", 1000)
        device_id = options.get("device_id", 0)
        model_dir = options.get("model_dir")
        model_filename = options.get("model_filename")
        params_filename = options.get("params_filename")
        model_precision = options.get("model_precision")
        use_mkldnn = options.get("use_mkldnn", False)
        use_mkldnnQuantizer = options.get("use_mkldnnQuantizer", False)
        use_iroptim = options.get("use_iroptim", True)
        cpu_num_thread = options.get("cpu_num_thread", 8)
        enable_memory_optim = options.get("enable_memory_optim", False)
        if model_filename and params_filename:
            prog_file = "%s/%s" % (model_dir, model_filename)
            params_file = "%s/%s" % (model_dir, params_filename)
            config = fluid.core.AnalysisConfig(prog_file, params_file)
        else:
            config = fluid.core.AnalysisConfig('{0}'.format(model_name))
            config.set_model(model_dir)
        if use_gpu:
            config.enable_use_gpu(gpu_memory, device_id)
            if enable_memory_optim:
                print("GPU memory optimization is enabled")
                config.enable_memory_optim(True)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(cpu_num_thread)
            if use_mkldnn:
                print("use_mkldnn")
                config.enable_mkldnn()
                if use_mkldnnQuantizer:
                    print("use_mkldnn_quantizer")
                    config.enable_quantizer()
            else:
                pass
        if use_tensorrt:
            print("use_tensorrt")
            if model_precision == "int8":
                print("use_tensorrt:int8")
                config.enable_tensorrt_engine(
                    1 << 20,
                    1,
                    use_calib_mode=True,
                    use_static=False,
                    precision_mode=fluid.core.AnalysisConfig.Precision.Int8)
            else:
                print("use_tensorrt:float")
                config.enable_tensorrt_engine(1 << 20, 1, use_static=False)
        if use_anakin:
            print("use_anakin")
            if model_precision == "int8":
                print("use_tensorrt:int8")
                config.enable_anakin_engine(
                    max_batch_size=1,
                    max_input_shape=self.max_input_shape,
                    precision_mode=fluid.core.AnalysisConfig.Precision.Int8,
                    auto_config_layout=self.auto_config_layout,
                    passes_filter=self.passes_filter,
                    ops_filter=self.ops_filter)
            else:
                print("use_tensorrt:float")
                config.enable_anakin_engine(
                    max_batch_size=1,
                    max_input_shape=self.max_input_shape,
                    auto_config_layout=self.auto_config_layout,
                    passes_filter=self.passes_filter,
                    ops_filter=self.ops_filter)
        config.switch_ir_optim(use_iroptim)
        self.config = config
        self.predictor = fluid.core.create_paddle_predictor(self.config)

    def run_inference(self, tensor, warmup=10, repeat_times=1000):
        """
        run repeat_times
        Args:
            tensor(list|PaddleTensor): list of Paddle Tensor
            warmup(int): warm-up times
            repeat_times(int): repeat times
        """
        for i in range(warmup):
            output = self.predictor.run(tensor)

        t = []
        for i in range(repeat_times):
            t1 = time.time()
            output = self.predictor.run(tensor)
            t2 = time.time()
            t.append((t2 - t1) * 1000)
        mean_t = np.mean(t)
        return mean_t

    def run(self, tensor):
        """
        run single time
        Args:
            tensor(list|PaddleTensor): list of Paddle Tensor
        """
        output = self.predictor.run(tensor)
        return output

    def run_exe(self, options, np_input):
        """
        run single time with executor
        Args:
            options(dict): dictionary it contains configurations
            np_input(numpy.array): numpy.array data with dtype
        """
        model_name = self.name
        use_gpu = options.get("use_gpu", False)
        model_dir = options.get("model_dir")
        if use_gpu:
            place = fluid.CUDAPlace()
        else:
            place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        [inference_program, feed_target_names, fetch_targets] = \
            fluid.io.load_inference_model(dirname=model_dir,
                                          executor=exe,
                                          model_filename='__model__',
                                          params_filename='__params__')
        test_program = inference_program.clone(for_test=True)
        result, = exe.run(
            test_program,
            feed={k: v
                  for k, v in zip(feed_target_names, np_input)},
            fetch_list=fetch_targets,
            return_numpy=False)
        return np.array(result)
