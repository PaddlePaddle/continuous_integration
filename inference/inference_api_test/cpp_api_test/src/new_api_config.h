// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <functional>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <utility>
#include <thread>
#include <map>
#include <chrono>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "include/paddle_inference_api.h"

DEFINE_string(model_path, "./mobilenetv1/model",
              "Directory of the infer model file.");
DEFINE_string(params_path, "", "Directory of the infer params file.");
DEFINE_string(model_name, "mobilenetv1", "name of model");
DEFINE_string(model_type, "static", "model generate type");
DEFINE_string(trt_precision, "fp32",
              "tensorrt precision type, choice = ['fp32', 'fp16', 'int8']");
DEFINE_string(image_shape, "3,224,224",
              "can only use for one input model(e.g. image classification)");

DEFINE_bool(use_gpu, false, "use_gpu or not");
DEFINE_bool(use_trt, false, "use trt or not");
DECLARE_bool(use_mkldnn, false, "use mkldnn or not");

DEFINE_int32(thread_num, 1, "num of threads");
DEFINE_int32(batch_size, 2, "batch size");
DEFINE_int32(repeats, 2, "repeats times");
DEFINE_int32(cpu_math_library_num_threads, 1, "cpu_math_library_num_threads");
DEFINE_int32(trt_min_subgraph_size, 3, "tensorrt min_subgraph_size");

DEFINE_double(accuracy, 1e-6, "Result Accuracy.");

std::map<std::string, paddle_infer::PrecisionType> trt_precision_map ={
  {"fp32", paddle_infer::PrecisionType::kFloat32},
  {"fp16", paddle_infer::PrecisionType::kHalf},
  {"int8", paddle_infer::PrecisionType::kInt8},
};

void PrepareConfig(paddle_infer::Config *config) {
  // prepare Paddle-Inference Config
  if (FLAGS_params_path != "") {
    LOG(INFO) << "params_path detected, set model with combined model";
    config->SetModel(FLAGS_model_path, FLAGS_params_path);
  }else {
    LOG(INFO) << "no params_path detected, set model with uncombined model";
    config->SetModel(FLAGS_model_path);
  }

  if (FLAGS_use_gpu) {
    config->EnableUseGpu(500, 0);
    if (FLAGS_use_trt) {
      bool use_calib = (trt_precision_map[FLAGS_trt_precision] == paddle_infer::PrecisionType::kInt8);
      config->EnableTensorRtEngine(1 << 30,  // workspace_size
          FLAGS_batch_size,  // max_batch_size
          FLAGS_trt_min_subgraph_size,  // min_subgraph_size
          trt_precision_map[FLAGS_trt_precision],  // Precision precision
          false,  // use_static
          use_calib);  //use_calib_mode
    }
  }else {
    config->DisableGpu();
    config->SetCpuMathLibraryNumThreads(FLAGS_cpu_math_library_num_threads);
    if (FLAGS_use_mkldnn) {
      config->EnableMKLDNN();
      LOG(INFO) << "mkldnn enabled";
    }
  }
  config->EnableMemoryOptim();
}

static void split(const std::string &str, const char *sep,
                  std::vector<std::string> *pieces, bool ignore_null = true) {
  pieces->clear();
  if (str.empty()) {
    if (!ignore_null) {
      pieces->push_back(str);
    }
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

// Compare result between two vector
template<typename T>
void CompareVectors(const std::vector<T> &outputs,
                    const std::vector<T> &ref_outputs) {
    EXPECT_EQ(outputs.size(), ref_outputs.size());
    for (size_t j = 0; j < outputs.size(); ++j) {
        EXPECT_LE(std::abs(outputs[j] - ref_outputs[j]), FLAGS_accuracy);
    }
}

template <typename... Args>
void SummaryConfig(paddle_infer::Config *config,
                   Args... kwargs){
  // TODO: apply Variadic template here

  LOG(INFO) << "----------------------- Model info ----------------------";
  LOG(INFO) << "Model name: " << FLAGS_model_name << ", " \
            << "Model type: " << FLAGS_model_type;
  LOG(INFO) << "----------------------- Data info -----------------------";
  LOG(INFO) << "Batch size: " << FLAGS_batch_size << ", " \
               "Num of samples: " << FLAGS_repeats;
  LOG(INFO) << "----------------------- Conf info -----------------------";
  LOG(INFO) << "device: " << (config->use_gpu() ? "gpu" : "cpu") << ", " \
                "ir_optim: " << (config->ir_optim() ? "true" : "false");
  LOG(INFO) << "enable_memory_optim: " << (config->enable_memory_optim() ? "true" : "false");
  if (config->use_gpu()) {
    LOG(INFO) << "enable_tensorrt: " << (config->tensorrt_engine_enabled() ? "true" : "false");
    if (config->tensorrt_engine_enabled()) {
      LOG(INFO) << "trt_precision: " << FLAGS_trt_precision;
    }
  }else {
    LOG(INFO) << "enable_mkldnn: " << (config->mkldnn_enabled() ? "true" : "false");
    LOG(INFO) << "cpu_math_library_num_threads: " << config->cpu_math_library_num_threads();
  }
  LOG(INFO) << "----------------------- Test info -----------------------";
  LOG(INFO) << "Test accuracy: " << FLAGS_accuracy;
  LOG(INFO) << "---------------------------------------------------------";
}
