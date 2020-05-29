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
    //config->EnableUseGpu(100, 0);
    //config->SwitchIrDebug(true);
    //config->SwitchIrOptim(true);
    config->SetModel(FLAGS_infer_model);
    //config->EnableTensorRtEngine();
}

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
    AnalysisConfig cfg;
    SetConfig(&cfg);
    if (use_mkldnn) {
        cfg.EnableMKLDNN();
        cfg.pass_builder()->AppendPass("fc_mkldnn_pass");
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
        config.pass_builder()->AppendPass("fc_mkldnn_pass");
    }
    std::vector<std::vector<PaddleTensor>> outputs;
    std::vector<std::vector<PaddleTensor>> inputs;
    LoadInputData(&inputs);
    TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&config),
                    inputs, &outputs, FLAGS_num_threads);
}

// compare native and analysis predictor
TEST(test_bert, compare_fluid) { compare(); }
// compare native and mkldnn
// TODO
/*
TEST(test_bert, compare_mkldnn) { compare(true, false); }
// compare native and ngraph
TEST(test_bert, compare_ngraph) { compare(false, true); }
*/
// multi thread 
TEST(test_bert, profile) { profile(); }
// multi thread with mkldnn
// TODO
/*
TEST(test_bert, profile_mkldnn) { profile(true, false); }
// multi thread with ngraph
TEST(Analyzer_bert, profile_ngraph) { profile(false, true); }
*/
}  // namespace test 
}  // namespace paddle

int main(int argc, char** argv) { 
    ::testing::InitGoogleTest(&argc, argv);
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS(); 
    //paddle::test::compare(false, true);
}
