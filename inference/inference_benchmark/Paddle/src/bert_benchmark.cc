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

#include <numeric>
#include <iostream>
#include <memory>
#include <sstream>

#include "./demo_helper.h"

namespace paddle_infer {

double Inference(Predictor* pred, int tid) {
  int batch_size = FLAGS_batch_size;  // int batch_size = 1;
  int seq_length = 128;
  // src_ids, int64[?,?,1]
  int input_num = batch_size * seq_length * 1;
  // pos_ids, int64[?,?,1]
  // sent_ids, int64[?,?,1]
  // input mask, float32[?,?,1]

  // prepare inputs
  std::vector<int64_t> src_ids_data(input_num);
  std::vector<int64_t> pos_ids_data(input_num);
  std::vector<int64_t> sent_ids_data(input_num);
  std::vector<float> mask_data(input_num);
  for (int i=0; i < input_num; ++i) {
    src_ids_data[i] = 1;
    pos_ids_data[i] = 1;
    sent_ids_data[i] = 1;
    mask_data[i] = 1.0;
  }

  auto in_names = pred->GetInputNames();
  // set inputs
  auto src_ids = pred->GetInputHandle(in_names[0]);
  src_ids->Reshape({batch_size, seq_length, 1});
  src_ids->CopyFromCpu(src_ids_data.data());

  auto pos_ids = pred->GetInputHandle(in_names[1]);
  pos_ids->Reshape({batch_size, seq_length, 1});
  pos_ids->CopyFromCpu(pos_ids_data.data());

  auto sent_ids = pred->GetInputHandle(in_names[2]);
  sent_ids->Reshape({batch_size, seq_length, 1});
  sent_ids->CopyFromCpu(sent_ids_data.data());

  auto mask = pred->GetInputHandle(in_names[3]);
  mask->Reshape({batch_size, seq_length, 1});
  mask->CopyFromCpu(mask_data.data());

  // warm-up
  for (size_t i = 0; i < FLAGS_warmup_times; ++i) {
    pred->Run();
    int out_num = 0;
    std::vector<float> out_data;
    auto out_names = pred->GetOutputNames();
    auto output_t = pred->GetOutputHandle(out_names[0]);

    std::vector<int> output_shape = output_t->shape();
    // retrive date to output vector
    out_num = std::accumulate(output_shape.begin(),
                              output_shape.end(), 1,
                              std::multiplies<int>());
    out_data.resize(out_num);
    output_t->CopyToCpu(out_data.data());
  }

  Timer pred_timer;  // init prediction timer
  int out_num = 0;
  std::vector<float> out_data;

  // main prediction process
  pred_timer.start();  // start timer
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    pred->Run();
    auto out_names = pred->GetOutputNames();
    auto output_t = pred->GetOutputHandle(out_names[0]);

    std::vector<int> output_shape = output_t->shape();
    // retrive date to output vector
    out_num = std::accumulate(output_shape.begin(),
                              output_shape.end(), 1,
                              std::multiplies<int>());
    out_data.resize(out_num);
    output_t->CopyToCpu(out_data.data());
  }
  pred_timer.stop();  // stop timer

  return pred_timer.report();
}

void DynamicShsapeConfig(Config *config) {
  const std::vector<int> min_shape = {1, 1, 1};
  const std::vector<int> max_shape = {static_cast<int>(FLAGS_batch_size),
                                      128, 1};
  const std::vector<int> opt_shape = {static_cast<int>(FLAGS_batch_size),
                                      64, 1};
  const std::vector<std::string> input_names = {"src_ids",
                                                "pos_ids",
                                                "sent_ids",
                                                "input_mask"};

  const std::map<std::string, std::vector<int>> min_input_shape = {
          {input_names[0], min_shape},
          {input_names[1], min_shape},
          {input_names[2], min_shape},
          {input_names[3], {1, 1, 1}},
  };
  const std::map<std::string, std::vector<int>> max_input_shape = {
          {input_names[0], max_shape},
          {input_names[1], max_shape},
          {input_names[2], max_shape},
          {input_names[3], {static_cast<int>(FLAGS_batch_size), 128, 1}},
  };
  const std::map<std::string, std::vector<int>> opt_input_shape = {
          {input_names[0], opt_shape},
          {input_names[1], opt_shape},
          {input_names[2], opt_shape},
          {input_names[3], {1, 128, 1}},
  };
  config->SetTRTDynamicShapeInfo(min_input_shape,
                                 max_input_shape,
                                 opt_input_shape);
  LOG(INFO) << "TensorRT dynamic shape enabled";
}

void RunDemo() {
  Config config;
  if (FLAGS_use_trt) {
    DynamicShsapeConfig(&config);
    // use for bert dynamic shape
    // if use tensorrt for bert model
    // need set dynamic shape first
  }
  PrepareConfig(&config);

  services::PredictorPool pred_pool(config, FLAGS_thread_num);

  auto total_time = Inference(pred_pool.Retrive(0), 0);

  SummaryConfig(&config, total_time);
}

}  // namespace paddle_infer


int main(int argc, char**argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  paddle_infer::RunDemo();
  return 0;
}

