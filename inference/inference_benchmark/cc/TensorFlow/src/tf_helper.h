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

#include <map>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <functional>

#include <stdlib.h>
#include <stdio.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/graph/default_device.h>
#include "tensorflow/c/c_api_experimental.h"

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

DEFINE_int32(thread_num, 1, "num of threads");
DEFINE_int32(batch_size, 1, "batch size");
DEFINE_int32(warmup_times, 10, "warmup times");
DEFINE_int32(repeats, 1000, "repeats times");

class Timer {
// Timer, count in ms
  public:
      Timer() {
          reset();
      }
      void start() {
          start_t = std::chrono::high_resolution_clock::now();
      }
      void stop() {
          auto end_t = std::chrono::high_resolution_clock::now();
          typedef std::chrono::microseconds ms;
          auto diff = end_t - start_t;
          ms counter = std::chrono::duration_cast<ms>(diff);
          total_time += counter.count();
      }
      void reset() {
          total_time = 0.;
      }
      double report() {
          return total_time / 1000.0;
      }
  private:
      double total_time;
      std::chrono::high_resolution_clock::time_point start_t;
};


template <typename... Args>
void SummaryConfig(double infer_time, Args... kwargs){
  // TODO: apply Variadic template here

  LOG(INFO) << "----------------------- Model info ----------------------";
  LOG(INFO) << "Model name: " << FLAGS_model_name << ", " \
            << "Model type: " << FLAGS_model_type;
  LOG(INFO) << "----------------------- Data info -----------------------";
  LOG(INFO) << "Batch size: " << FLAGS_batch_size << ", " \
               "Num of samples: " << FLAGS_repeats;
  LOG(INFO) << "----------------------- Conf info -----------------------";
  char *TF_XLA_FLAGS;
  TF_XLA_FLAGS = getenv("TF_XLA_FLAGS");
  LOG(INFO) << "TF_XLA_FLAGS=" << TF_XLA_FLAGS;
  LOG(INFO) << "TF_XLA_FLAGS enabled: " << (TF_XLA_FLAGS ? "true" : "false");
  LOG(INFO) << "----------------------- Perf info -----------------------";
  LOG(INFO) << "Average latency(ms): " << infer_time / FLAGS_repeats << ", " \
            << "QPS: " << (FLAGS_repeats * FLAGS_batch_size)/ (infer_time/1000);
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