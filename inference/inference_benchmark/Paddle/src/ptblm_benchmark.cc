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

#include "demo_helper.h"

namespace paddle_infer {

double Inference(Predictor* pred, int tid) {
  int batch_size = FLAGS_batch_size;
  // init num_step
  int num_step = 20;
  int input_num = batch_size * num_step;

  int64_t *input = new int64_t[input_num];
  memset(input, 0, input_num * sizeof(int64_t));

  // init init_hidden
  int num_layer = 2;
  int hidden_size = 200;

  int hidden_num = num_layer * hidden_size * batch_size;
  float *init_hidden = new float[hidden_num];
  memset(init_hidden, 0, hidden_num * sizeof(float));

  // init init_hidden
  float *init_cell = new float[hidden_num];
  memset(init_cell, 0, hidden_num * sizeof(float));

  // set inputs
  auto in_names = pred->GetInputNames();
  auto input_t = pred->GetInputHandle(in_names[0]); // input
  input_t->Reshape({batch_size, num_step});
  input_t->CopyFromCpu(input);

  // set input init hidden
  auto init_hidden_t = pred->GetInputHandle(in_names[1]); // init_hidden
  init_hidden_t->Reshape({num_layer, batch_size, hidden_size});
  init_hidden_t->CopyFromCpu(init_hidden);

  // set input init cell
  auto init_cell_t = pred->GetInputHandle(in_names[2]);  // init_cell
  init_cell_t->Reshape({num_layer, batch_size, hidden_size});
  init_cell_t->CopyFromCpu(init_cell);

  // warm-up
  for (size_t i = 0; i < FLAGS_warmup_times; ++i) {
    pred->Run();
    int out_num = 0;
    std::vector<float> out_data;
    auto out_names = pred->GetOutputNames();
    auto output_t = pred->GetOutputHandle(out_names[0]);

    std::vector<int> output_shape = output_t->shape();
    // retrive date to output vector
    out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    out_data.resize(out_num);
    output_t->CopyToCpu(out_data.data());
  }

  Timer pred_timer; // init prediction timer
  int out_num = 0;
  std::vector<float> out_data;

  // main prediction process
  pred_timer.start(); // start timer
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    pred->Run();
    auto out_names = pred->GetOutputNames();
    auto output_t = pred->GetOutputHandle(out_names[0]);

    std::vector<int> output_shape = output_t->shape();
    // retrive date to output vector
    out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    out_data.resize(out_num);
    output_t->CopyToCpu(out_data.data());
  }
  pred_timer.stop(); // stop timer

  return pred_timer.report();
}

void RunDemo() {
  Config config;
  PrepareConfig(&config);

  services::PredictorPool pred_pool(config, FLAGS_thread_num);

  auto total_time = Inference(pred_pool.Retrive(0), 0);

  SummaryConfig(&config, total_time);
}

} // namespace paddle_infer


int main(int argc, char**argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  paddle_infer::RunDemo();
  return 0;
}

