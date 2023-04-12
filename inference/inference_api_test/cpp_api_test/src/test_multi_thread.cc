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
#include <thread> // NOLINT
#include <fstream>
#include <map>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "include/paddle_inference_api.h"

#include "test_helper.h" //NOLINT

DEFINE_int32(paddle_num_threads, 1, "Number of threads for CPU inference")

namespace paddle {
namespace test {

using paddle::AnalysisConfig;

void SetConfig(AnalysisConfig *cfg,
               int batch_size,
               bool use_gpu,
               bool use_trt,
               bool use_mkldnn) {
    cfg->SetModel(FLAGS_infer_model + "/__model__",
                  FLAGS_infer_model + "/__params__");
    if (use_gpu) {
        cfg->EnableUseGpu(100, 0);
        if (use_trt) {
            cfg->EnableTensorRtEngine(1 << 20,
                                      batch_size, 3,
                                      AnalysisConfig::Precision::kFloat32,
                                      false);
        }
    }
    cfg->SwitchIrOptim();
    cfg->SwitchSpecifyInputNames();
    if (!use_mkldnn) {
        cfg->EnableMemoryOptim();
    }
    if (!use_gpu) {
        if (use_mkldnn) {
            cfg->SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);
            cfg->EnableMKLDNN();
        }
    }
    cfg->SwitchUseFeedFetchOps(false);
}

bool test_multi_threads(int batch_size, int repeat) {
    // set cpu config
    AnalysisConfig config0;
    SetConfig(&config0, batch_size,
              false, false, false);

    // set gpu config
    AnalysisConfig config1;
    SetConfig(&config1, batch_size,
              true, false, false);

    // create paddle predictor for each config
    auto pred0 = CreatePaddlePredictor(config0);
    auto pred1 = CreatePaddlePredictor(config1);

    PaddlePredictor *pres[3];
    pres[0] = pred0.get();
    pres[1] = pred1.get();

    int channels = 3;
    int height = 224;
    int width = 224;
    int input_num = channels * height * width * batch_size;

    // prepare inputs
    float *input = new float[input_num];
    memset(input, 0, input_num * sizeof(float));

    // total_outputs structure
    // [ output of pres[0], output of pres[1] ]
    std::vector<std::vector<std::vector<float>>> total_outputs;
    std::vector<std::thread> threads;
    int num_jobs = 2;
    for (int tid = 0; tid < num_jobs; tid++) {
        threads.emplace_back([&, tid](){
            std::vector<std::vector<float>> local_outputs;
            for (size_t i = 0; i < repeat; i++) {
                // Prepare input
                auto input_names = pres[tid]->GetInputNames();
                auto input_t = pres[tid]->GetInputTensor(input_names[0]);
                input_t->Reshape({batch_size, channels, height, width});
                input_t->copy_from_cpu(input);

                // run
                pres[tid]->ZeroCopyRun();

                // Get output
                std::vector<float> out_data;
                auto output_names = pres[tid]->GetOutputNames();
                auto output_t = pres[tid]->GetOutputTensor(output_names[0]);
                std::vector<int> output_shape = output_t->shape();
                int out_num = std::accumulate(output_shape.begin(),
                                              output_shape.end(), 1,
                                              std::multiplies<int>());

                out_data.resize(out_num);
                output_t->copy_to_cpu(out_data.data());

                local_outputs.emplace_back(out_data);
            }
            total_outputs.emplace_back(local_outputs);
        });
    }

    for (int i = 0; i < num_jobs; i++) {
        threads[i].join();
    }
    LOG(INFO) << total_outputs.size();
    LOG(INFO) << total_outputs[0].size();
    CompareVectors(total_outputs[0][0], total_outputs[1][0]);
    return true;
}

TEST(test_multi_threads, compare_threads) {
    // each thread has one configuration
    test_multi_threads(1, 100);
    }

}  // namespace test
}  // namespace paddle

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS();
}
