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

#include <algorithm>
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <thread>  // NOLINT
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle_inference_api.h"
#include "test_helper.h"
#include "bert_test_helper.h"

namespace paddle {
namespace test {
void SetConfig(AnalysisConfig *config) {
    config->SetModel(FLAGS_infer_model);
    if (FLAGS_use_gpu) {
        config->EnableUseGpu(100, 0);
    } else {
        config->DisableGpu();
    }
    // config->SwitchIrDebug(true);
    // config->SwitchIrOptim(true);
    // config->EnableTensorRtEngine();
}

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
    AnalysisConfig cfg;
    SetConfig(&cfg);
    if (FLAGS_use_gpu) {
        cfg.EnableMemoryOptim();
    } else {
        if (use_mkldnn) {
            cfg.EnableMKLDNN();
            cfg.SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);
        }
    }
    std::vector<std::vector<PaddleTensor>> inputs;
    LoadInputData(&inputs);
    CompareNativeAndAnalysis(
        reinterpret_cast<const PaddlePredictor::Config *>(&cfg), inputs);
}

void profile(bool use_mkldnn = false) {
    AnalysisConfig config;
    SetConfig(&config);
    if (use_mkldnn) {
        config.EnableMKLDNN();
    }
    std::vector<std::vector<PaddleTensor>> outputs;
    std::vector<std::vector<PaddleTensor>> inputs;
    LoadInputData(&inputs);
    TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&config),
                    inputs, &outputs, FLAGS_num_threads);
}

// compare native and analysis predictor
TEST(test_bert, compare_fluid) { compare(); }

// multi thread
TEST(test_bert, profile) { profile(); }

}  // namespace test
}  // namespace paddle

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS();
    // paddle::test::compare(false, true);
}

