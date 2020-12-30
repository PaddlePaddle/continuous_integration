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

std::vector<int64_t> Inference(Predictor* pred, int tid) {
  int channels = 3;
  int height = 640;
  int width = 640;
  int batch_size = FLAGS_batch_size;
  int input_num = channels * height * width * batch_size;

  // init im_info
  int info_num = 3 * batch_size;
  std::vector<int> im_info_size(info_num);
  for (int i=0; i < batch_size; ++i) {
    im_info_size[i] = 640;
    im_info_size[i + 1] = 640;
    im_info_size[i + 2] = 3;
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

  // set input im_info
  // im_info, int32[?,3]
  auto info_t = pred->GetInputHandle(in_names[1]);
  info_t->Reshape({batch_size, 3});
  info_t->CopyFromCpu(im_info_size.data());

  // NOTICE: solov2 model outputs
  // 0, id: save_infer_model/scale_0.tmp_0, int64[?]
  // 1, id: save_infer_model/scale_1.tmp_0, float32[?]
  // 2, id: save_infer_model/scale_2.tmp_0, int32[?,?,?]
  int out_num = 0;
  std::vector<int64_t> out_data;
  // main prediction process
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    EXPECT_TRUE(pred->Run());
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

TEST(test_solo_model, ir_compare) {
    // Test solo model
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
