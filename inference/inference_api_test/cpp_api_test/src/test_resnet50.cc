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

#include <fstream>
#include <iostream>

#include "gtest/gtest.h"
#include "test_helper.h" //NOLINT
#include <gflags/gflags.h>

DEFINE_bool(disable_mkldnn_fc, false, "Disable usage of MKL-DNN's FC op");
DEFINE_int32(paddle_num_threads, 1, "Number of threads for CPU inference");

namespace paddle {
namespace test {

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/__model__",
          FLAGS_infer_model + "/__params__");
  if (FLAGS_use_gpu) {
      cfg->EnableUseGpu(100, 0);
  }
  cfg->SwitchIrOptim();
  cfg->SwitchSpecifyInputNames();
  if (!FLAGS_test_mkldnn) {
      cfg->EnableMemoryOptim();
  }
  if (!FLAGS_use_gpu) {
      if (FLAGS_test_mkldnn) {
          cfg->SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);
          cfg->EnableMKLDNN();
      }
  }
}

template <typename T>
class TensorReader {
 public:
  TensorReader(std::ifstream &file, size_t beginning_offset,
               std::vector<int> shape, std::string name)
      : file_(file), position(beginning_offset), shape_(shape), name_(name) {
    numel = std::accumulate(shape_.begin(), shape_.end(), size_t{1},
                            std::multiplies<size_t>());
  }

  PaddleTensor NextBatch() {
    PaddleTensor tensor;
    tensor.name = name_;
    tensor.shape = shape_;
    tensor.dtype = GetPaddleDType<T>();
    tensor.data.Resize(numel * sizeof(T));

    file_.seekg(position);
    file_.read(static_cast<char *>(tensor.data.data()), numel * sizeof(T));
    position = file_.tellg();

    if (file_.eof()) LOG(ERROR) << name_ << ": reached end of stream";
    if (file_.fail())
      throw std::runtime_error(name_ + ": failed reading file.");

    return tensor;
  }

 protected:
  std::ifstream &file_;
  size_t position;
  std::vector<int> shape_;
  std::string name_;
  size_t numel;
};

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs,
              int32_t batch_size = FLAGS_batch_size) {
  std::ifstream file(FLAGS_infer_data, std::ios::binary);
  if (!file) {
    FAIL() << "Couldn't open file: " << FLAGS_infer_data;
  }

  int64_t total_images{0};
  file.read(reinterpret_cast<char *>(&total_images), sizeof(total_images));
  LOG(INFO) << "Total images in file: " << total_images;

  std::vector<int> image_batch_shape{batch_size, 3, 224, 224};
  std::vector<int> label_batch_shape{batch_size, 1};
  auto images_offset_in_file = static_cast<size_t>(file.tellg());
  auto labels_offset_in_file =
      images_offset_in_file + sizeof(float) * total_images * 3 * 224 * 224;

  TensorReader<float> image_reader(file, images_offset_in_file,
                                   image_batch_shape, "image");
  TensorReader<int64_t> label_reader(file, labels_offset_in_file,
                                     label_batch_shape, "label");

  auto iterations_max = total_images / batch_size;
  auto iterations = iterations_max;
  if (FLAGS_iterations > 0 && FLAGS_iterations < iterations_max) {
    iterations = FLAGS_iterations;
  }
  for (auto i = 0; i < iterations; i++) {
    auto images = image_reader.NextBatch();
    auto labels = label_reader.NextBatch();
    inputs->emplace_back(std::vector<PaddleTensor>{std::move(images)});
  }
}

// Compare result of NativeConfig and AnalysisConfig
void compare(bool test_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  if (test_mkldnn) {
    cfg.EnableMKLDNN();
    if (!FLAGS_disable_mkldnn_fc)
      cfg.pass_builder()->AppendPass("fc_mkldnn_pass");
  }

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

void CompareOptimAndOrig(const PaddlePredictor::Config *orig_config,
                         const PaddlePredictor::Config *optim_config,
                         const std::vector<std::vector<PaddleTensor>> &inputs) {
  PrintConfig(orig_config, true);
  PrintConfig(optim_config, true);
  std::vector<std::vector<PaddleTensor>> orig_outputs, optim_outputs;
  TestOneThreadPrediction(orig_config, inputs, &orig_outputs, false);
  TestOneThreadPrediction(optim_config, inputs, &optim_outputs, false);
  CompareResult(orig_outputs.back(), optim_outputs.back());
}

TEST(test_resnet50, compare) { compare(); }

TEST(test_resnet50, compare_mkldnn) { compare(true); }

TEST(test_resnet50, compare_use_gpu) {
  AnalysisConfig cfg;
  cfg.SetModel(FLAGS_infer_model + "/__model__",
          FLAGS_infer_model + "/__params__");
  cfg.EnableUseGpu(100, 0);
  cfg.SwitchIrOptim();
  cfg.SwitchSpecifyInputNames();
  cfg.EnableMemoryOptim();
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

TEST(test_resnet50, compare_disable_gpu) {
  AnalysisConfig cfg;
  cfg.SetModel(FLAGS_infer_model + "/__model__",
          FLAGS_infer_model + "/__params__");
  cfg.DisableGpu();
  ASSERT_FALSE(cfg.use_gpu());
  cfg.SwitchIrOptim();
  cfg.SwitchSpecifyInputNames();
  cfg.EnableMemoryOptim();
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

TEST(test_resnet50, use_static_optim) {
  AnalysisConfig cfg;
  cfg.SetModel(FLAGS_infer_model + "/__model__",
          FLAGS_infer_model + "/__params__");
  cfg.EnableUseGpu(100, 0);
  cfg.SwitchIrOptim();
  cfg.SwitchSpecifyInputNames();
  cfg.EnableMemoryOptim();
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

}  // namespace test
}  // namespace paddle



int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS();
}
