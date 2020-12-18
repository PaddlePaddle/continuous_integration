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

#include "./new_api_config.h"

namespace paddle_infer {

std::vector<float> Inference(Predictor* pred, int tid) {
  // parse FLAGS_image_shape to vector
  std::vector<std::string> shape_strs;
  split(FLAGS_image_shape, ",", &shape_strs, true);

  // int channels = 3;
  int channels = static_cast<int>(std::stoi(shape_strs[0]));
  // int height = 224;
  int height = static_cast<int>(std::stoi(shape_strs[1]));
  // int width = 224;
  int width = static_cast<int>(std::stoi(shape_strs[2]));

  int batch_size = FLAGS_batch_size;  // int batch_size = 1;
  int input_num = channels * height * width * batch_size;
  LOG(INFO) << "batch_size: " << batch_size << ",\t" \
            << "channels: " << channels << ",\t" \
            << "height: " << height << ",\t" \
            << "width: " << width;

  // prepare inputs
  std::vector<float> in_data(input_num);
  for (int i=0; i < input_num; ++i) {
    in_data[i] = i % 10 * 0.1;
  }

  // set inputs
  auto in_names = pred->GetInputNames();
  auto input_t = pred->GetInputHandle(in_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->CopyFromCpu(in_data.data());

  int out_num = 0;
  std::vector<float> out_data;
  // main prediction process
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

  return out_data;
}

TEST(test_pdclas_model, ir_compare) {
    // Test PaddleClas model
    // ir optim compare

    Config config;
    PrepareConfig(&config);
    services::PredictorPool pred_pool(config, 1);
    auto out_data1 = Inference(pred_pool.Retrive(0), 0);

    Config no_ir_config;
    PrepareConfig(&no_ir_config);
    no_ir_config.SwitchIrOptim(false); 
    services::PredictorPool pred_pool2(no_ir_config, 1);
    auto out_data2 = Inference(pred_pool2.Retrive(0), 0);

    SummaryConfig(&config);
    CompareVectors(out_data1, out_data2);
}

}  // namespace paddle_infer


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS();
}
