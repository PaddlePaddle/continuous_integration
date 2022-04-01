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
#include <memory>

#include "./demo_helper.h"

namespace paddle_infer {

template<typename T = float>
double Inference(Predictor* pred, int tid) {
  Timer pred_timer;
  std::vector<int> index(1000);
  float test_num = 0;
  float top1_num = 0;
  float top5_num = 0;

  // parse FLAGS_image_shape to vector
  std::vector<std::string> shape_strs;
  split(FLAGS_image_shape, ",", &shape_strs);

  // int channels = 3;
  int channels = static_cast<int>(std::stoi(shape_strs[0]));
  // int height = 224;
  int height = static_cast<int>(std::stoi(shape_strs[1]));
  // int width = 224;
  int width = static_cast<int>(std::stoi(shape_strs[2]));

  int batch_size = FLAGS_batch_size;  // int batch_size = 1;
  int input_num = channels * height * width * batch_size;
  LOG(INFO) << "batch_size: " << batch_size << "\t," \
            << "channels: " << channels << "\t," \
            << "height: " << height << "\t," \
            << "width: " << width;

  std::vector<T> out_data;
  int out_num = 0;
  // read imagenet-eval-binary data
  for (size_t ind = 0; ind < 50000; ind++) {
    // load data to vector
    std::vector<float> in_data(input_num); // total batch image vector
    int label = 0;
    std::string data_path = FLAGS_binary_data_path + "/" + std::to_string(ind) + ".data";
    LoadBinaryData(data_path.c_str(), &in_data, label, input_num);
    // verfiy load data correct
    // for (int i=0; i < 5; ++i){
    //   LOG(INFO) << "input : " << in_data[i]; 
    // }

    // set inputs
    auto in_names = pred->GetInputNames();
    auto input_t = pred->GetInputHandle(in_names[0]);
    input_t->Reshape({batch_size, channels, height, width});
    input_t->CopyFromCpu(in_data.data());

    pred_timer.start(); // start counter
    // predictions
    CHECK(pred->Run());
    auto out_names = pred->GetOutputNames();
    auto output_t = pred->GetOutputHandle(out_names[0]);
    std::vector<int> output_shape = output_t->shape();
    // retrive date to output vector
    out_num = std::accumulate(output_shape.begin(),
                              output_shape.end(), 1,
                              std::multiplies<int>());
    out_data.resize(out_num);
    output_t->CopyToCpu(out_data.data());
    pred_timer.stop(); // stop counter

    // calculate accuray of model
    std::iota(index.begin(), index.end(), 0);
    std::sort(index.begin(), index.end(), [out_data](size_t i1, size_t i2) {
      return out_data[i1] > out_data[i2];
    });
    test_num++;
    if (label == index[0]) {
      top1_num++;
    }
    for (int i = 0; i < 5; i++) {
      if (label == index[i]) {
        top5_num++;
      }
    }
  }
  LOG(INFO) << "=== accuracy final result ===";
  LOG(INFO) << "top1 acc: " << top1_num / test_num;
  LOG(INFO) << "top5 acc: " << top5_num / test_num;

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

