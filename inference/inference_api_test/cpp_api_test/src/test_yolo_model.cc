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
  int channels = 3;
  int height = 608;
  int width = 608;
  int batch_size = FLAGS_batch_size;
  int input_num = channels * height * width * batch_size;

  // init im_shape
  int shape_num = 2 * batch_size;
  // int *shape = new int[shape_num];
  // memset(shape, height, shape_num * sizeof(int));
  std::vector<int> shape(shape_num);
  for (int i=0; i < batch_size; ++i) {
    shape[i] = 608;
    shape[i + 1] = 608;
  }

  // prepare inputs
  std::vector<float> in_data(input_num);
  for (int i=0; i < input_num; ++i) {
    in_data[i] = i % 10 * 0.1;
  }

  // set inputs
  // image, float32[?,3,608,608]
  auto in_names = pred->GetInputNames();
  auto input_t = pred->GetInputHandle(in_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->CopyFromCpu(in_data.data());

  // set input data shape
  // im_size, int32[?,2]
  auto shape_t = pred->GetInputHandle(in_names[1]);
  shape_t->Reshape({batch_size, 2});
  shape_t->CopyFromCpu(shape.data());

  int out_num = 0;
  std::vector<float> out_data;
  // main prediction process
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    pred->Run();
    auto out_names = pred->GetOutputNames();
    auto output_t = pred->GetOutputHandle(out_names[0]);

    std::vector<int> output_shape = output_t->shape();
    // retrieve date to output vector
    out_num = std::accumulate(output_shape.begin(),
                              output_shape.end(), 1,
                              std::multiplies<int>());
    out_data.resize(out_num);
    output_t->CopyToCpu(out_data.data());
  }

  return out_data;
}

TEST(test_yolo_model, ir_compare) {
    // Test PaddleClas model
    // ir optim compare

    Config config;
    PrepareConfig(&config);
    services::PredictorPool pred_pool(config, 1);
    auto out_data1 = Inference(pred_pool.Retrieve(0), 0);

    Config no_ir_config;
    PrepareConfig(&no_ir_config);
    no_ir_config.SwitchIrOptim(false); 
    services::PredictorPool pred_pool2(no_ir_config, 1);
    auto out_data2 = Inference(pred_pool2.Retrieve(0), 0);

    SummaryConfig(&config);
    CompareVectors(out_data1, out_data2);
}

}  // namespace paddle_infer


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS();
}
