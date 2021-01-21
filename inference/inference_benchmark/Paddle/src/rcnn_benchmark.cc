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
  // parse FLAGS_image_shape to vector
  std::vector<std::string> shape_strs;
  split(FLAGS_image_shape, ",", &shape_strs);

  int channels = static_cast<int>(std::stoi(shape_strs[0]));  // int channels = 3;
  int height = static_cast<int>(std::stoi(shape_strs[1]));  // int height = 224;
  int width = static_cast<int>(std::stoi(shape_strs[2]));  // int width = 224;

  int batch_size = FLAGS_batch_size;  // int batch_size = 1;
  int input_num = channels * height * width * batch_size;
  LOG(INFO) << "batch_size: " << batch_size << "\t," \
            << "channels: " << channels << "\t," \
            << "height: " << height << "\t," \
            << "width: " << width;

  // prepare inputs
  std::vector<float> in_data(input_num);
  for (int i=0; i < input_num; ++i) {
    in_data[i] = i % 10 * 0.1;
  }

  int shape_num = batch_size * channels;
  std::vector<float> shape_data(shape_num);
  for (int i=0; i < batch_size; ++i){
    shape_data[i] = 640;
    shape_data[i + 1] = 640;
    shape_data[i + 2] = 1;
  }

  std::vector<float> info_data(shape_num);
  for (int i=0; i < batch_size; ++i) {
    info_data[i] = 480;
    info_data[i + 1] = 640;
    info_data[i + 2] = 1;
  }

  auto in_names = pred->GetInputNames();
  // set inputs image, float32[?,3,640,640]
  auto input_t = pred->GetInputHandle(in_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->CopyFromCpu(in_data.data());

  // set image shape, float32[?,3]
  auto shape_t = pred->GetInputHandle(in_names[1]);
  shape_t->Reshape({batch_size, channels});
  shape_t->CopyFromCpu(shape_data.data());

  // set image info, float32[?,3]
  auto info_t = pred->GetInputHandle(in_names[2]);
  info_t->Reshape({batch_size, channels});
  info_t->CopyFromCpu(info_data.data());


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

void RunDemo() {
  Config config;
  PrepareConfig(&config);

  services::PredictorPool pred_pool(config, FLAGS_thread_num);

  auto total_time = Inference(pred_pool.Retrive(0), 0);

  SummaryConfig(&config, total_time);
}

}  // namespace paddle_infer


int main(int argc, char**argv) {
  ::GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  paddle_infer::RunDemo();
  return 0;
}

