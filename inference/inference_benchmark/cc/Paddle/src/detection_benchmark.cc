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
  // Detection model benchmark test
  // use for Paddle-2.0 saved Paddle-Detection inference model

  // parse input tensor shape
  std::string shape_line;
  std::ifstream shape_file(FLAGS_txt_image_shape_path.c_str());
    if (!shape_file.is_open()) {
      LOG(FATAL) << "open input file fail.";
  }
  std::getline(shape_file, shape_line);
  // parse input shape to vector, split by space
  std::vector<std::string> shape_strs;
  split(shape_line, " ", &shape_strs);

  int batch_size = FLAGS_batch_size;  // int batch_size = stoi(shape_strs[0]);
  // calculate elements of input image
  int input_num=batch_size;
  for (int j=0; j<shape_strs.size();j++){
    std::cout << "shape_strs : " << shape_strs[j] << std::endl;
    input_num*=stoi(shape_strs[j]);
  }
  shape_file.close();

  int channels = stoi(shape_strs[1]);
  int height = stoi(shape_strs[2]);
  int width = stoi(shape_strs[3]);

  LOG(INFO) << "----------------------- Input info ----------------------";
  LOG(INFO) << "Input image shape: " << channels << ", " << height << ", " << width;
  LOG(INFO) << "Total input digits num: " << input_num;
  LOG(INFO) << "---------------------------------------------------------";

  int shape_num = batch_size * 2;
  std::vector<float> shape_data(shape_num);
  LOG(INFO) << "Total shape_num: " << shape_num;
  for (int i=0; i < batch_size; ++i){
    shape_data[i] = stoi(shape_strs[2]);  // shape_data[i] = 800;
    shape_data[i + 1] = stoi(shape_strs[3]);  // shape_data[i + 1] = 1200;
  }

  std::vector<float> scale_factor(shape_num);
  for (int i=0; i < batch_size; ++i) {
    scale_factor[i] = 1.0;
    scale_factor[i + 1] = 1.0;
  }

  // read data from txt file
  std::vector<float> one_image(input_num/batch_size); // only one image for one txt
  std::vector<float> in_data(input_num); // total batch image vector
  LoadTxtImageData(FLAGS_txt_image_path.c_str(), &in_data);
  for (int i=0; i < batch_size; ++i) {
    // concat for several batch_size
    in_data.insert( in_data.end(), one_image.begin(), one_image.end() );
  }  
  // in_data.insert(in_data.end(), in_data.begin(), in_data.end()); // concat for several batch_size

  auto in_names = pred->GetInputNames();
  // set image shape, float32[?,2]
  auto shape_t = pred->GetInputHandle(in_names[0]);
  shape_t->Reshape({batch_size, 2});
  shape_t->CopyFromCpu(shape_data.data());

  // set inputs image, float32[?,3,x,x]
  auto input_t = pred->GetInputHandle(in_names[1]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->CopyFromCpu(in_data.data());

  // set scale_factor, float32[?,2]
  auto info_t = pred->GetInputHandle(in_names[2]);
  info_t->Reshape({batch_size, 2});
  info_t->CopyFromCpu(scale_factor.data());

  // warm-up
  for (size_t i = 0; i < FLAGS_warmup_times; ++i) {
    pred->Run();
    int out_num = 0;
    // std::vector<float> out_data;
    std::vector<T> out_data;
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
  // std::vector<float> out_data;
  std::vector<T> out_data;

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
  auto trt_min_shape = {1,3, 320, 320};
  auto trt_max_shape = {1, 3, 1280, 1280};
  auto trt_opt_shape = {1, 3, 960, 960};

  // set DynamicShsape for image tensor
  const std::map<std::string, std::vector<int>> min_input_shape = {{"image", trt_min_shape}};
  const std::map<std::string, std::vector<int>> max_input_shape = {{"image", trt_max_shape}};
  const std::map<std::string, std::vector<int>> opt_input_shape = {{"image", trt_opt_shape}};

  config->SetTRTDynamicShapeInfo(min_input_shape,
                                 max_input_shape,
                                 opt_input_shape);
  LOG(INFO) << "TensorRT dynamic shape enabled";
}

void RunDemo() {
  Config config;

  // dy2static detection model need to use this feature
  if (FLAGS_use_trt) {
    // use for trt dynamic shape only
    DynamicShsapeConfig(&config);
  }
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
