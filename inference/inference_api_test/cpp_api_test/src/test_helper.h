#pragma once

#include <algorithm>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <functional>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <thread>  // NOLINT

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle_inference_api.h"


DEFINE_string(model_name, "", "model name");
DEFINE_string(infer_model, "", "model path");
DEFINE_string(infer_data, "", "data file");
DEFINE_string(refer_result, "", "reference result for comparison");
DEFINE_int32(batch_size, 1, "batch size");
// DEFINE_int32(warmup_batch_size, 100, "batch size for quantization warmup");
// setting iterations to 0 means processing the whole dataset
DEFINE_int32(iterations, 0, "number of batches to process");
DEFINE_int32(repeat, 1, "Running the inference program repeat times.");
DEFINE_bool(test_all_data, false, "Test the all dataset in data file.");
DEFINE_int32(num_threads, 1, "Running the inference program in multi-threads.");
DEFINE_bool(use_analysis, true,
            "Running the inference program in analysis mode.");
// DEFINE_bool(record_benchmark, false,
//             "Record benchmark after profiling the model");
DEFINE_double(accuracy, 1e-3, "Result Accuracy.");
DEFINE_double(quantized_accuracy, 1e-2, "Result Quantized Accuracy.");
DEFINE_bool(zero_copy, false, "Use ZeroCopy to speedup Feed/Fetch.");
DEFINE_bool(warmup, false,
            "Use warmup to calculate elapsed_time more accurately. "
            "To reduce CI time, it sets false in default.");
DEFINE_bool(use_gpu, false, "Whether use gpu.");
DEFINE_bool(use_trt, false, "Whether use trt.");
DEFINE_bool(use_mkldnn, false, "Whether use mkldnn.");

DECLARE_bool(profile);
DECLARE_int32(paddle_num_threads);


namespace paddle {
namespace test {

using paddle::PaddleTensor;

template <typename T>
int VecReduceToInt(const std::vector<T> &v) {
      return std::accumulate(v.begin(), v.end(), 1, [](T a, T b) { return a * b; });
}

template <typename T>
    constexpr paddle::PaddleDType GetPaddleDType();

template <>
    constexpr paddle::PaddleDType GetPaddleDType<int64_t>() {
          return paddle::PaddleDType::INT64;
    }

template <>
    constexpr paddle::PaddleDType GetPaddleDType<float>() {
          return paddle::PaddleDType::FLOAT32;
    }


template <typename T>
void GetValueFromStream(std::stringstream *ss, T *t) {
  (*ss) >> (*t);
}

template <>
void GetValueFromStream<std::string>(std::stringstream *ss, std::string *t) {
  *t = ss->str();
}

// Split string to vector
template <typename T>
void Split(const std::string &line, char sep, std::vector<T> *v) {
  std::stringstream ss;
  T t;
  for (auto c : line) {
    if (c != sep) {
      ss << c;
    } else {
      GetValueFromStream<T>(&ss, &t);
      v->push_back(std::move(t));
      ss.str({});
      ss.clear();
    }
  }
  if (!ss.str().empty()) {
    GetValueFromStream<T>(&ss, &t);
    v->push_back(std::move(t));
    ss.str({});
    ss.clear();
  }
}


static void split(const std::string &str, char sep,
                  std::vector<std::string> *pieces, bool ignore_null = true) {
  pieces->clear();
  if (str.empty()) {
    if (!ignore_null) {
      pieces->push_back(str);
    }
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

template <typename T>
static T convert(const std::string &item,
                 std::function<T(const std::string &item)> func) {
  T res;
  try {
    res = func(item);
  } catch (std::invalid_argument &e) {
    std::string message =
        "invalid_argument exception when try to convert : " + item;
    LOG(ERROR) << message;
    //PADDLE_THROW(message);
  } catch (std::out_of_range &e) {
    std::string message =
        "out_of_range exception when try to convert : " + item;
    LOG(ERROR) << message;
    //PADDLE_THROW(message);
  } catch (...) {
    std::string message = "unexpected exception when try to convert " + item;
    LOG(ERROR) << message;
    //PADDLE_THROW(message);
  }
  return res;
}


static void split_to_int64(const std::string &str, char sep,
                           std::vector<int64_t> *is) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(), pieces.end(), std::back_inserter(*is),
                 [](const std::string &v) {
                   return convert<int64_t>(v, [](const std::string &item) {
                     return std::stoll(item);
                   });
                 });
}

template <typename T>
std::string to_string(const std::vector<T> &vec) {
  std::stringstream ss;
  for (const auto &c : vec) {
    ss << c << " ";
  }
  return ss.str();
}

template <typename T>
static void ZeroCopyTensorAssignData(ZeroCopyTensor *tensor,
                                     const PaddleBuf &data) {
  auto *ptr = tensor->mutable_data<T>(PaddlePlace::kCPU);
  for (size_t i = 0; i < data.length() / sizeof(T); i++) {
    ptr[i] = *(reinterpret_cast<T *>(data.data()) + i);
  }
}

void ConvertPaddleTensorToZeroCopyTensor(
    PaddlePredictor *predictor, const std::vector<PaddleTensor> &inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    auto tensor = predictor->GetInputTensor(input.name);
    tensor->Reshape(input.shape);
    tensor->SetLoD({input.lod});
    if (input.dtype == PaddleDType::INT64) {
      ZeroCopyTensorAssignData<int64_t>(tensor.get(), input.data);
    } else if (input.dtype == PaddleDType::FLOAT32) {
      ZeroCopyTensorAssignData<float>(tensor.get(), input.data);
    } else if (input.dtype == PaddleDType::INT32) {
      ZeroCopyTensorAssignData<int32_t>(tensor.get(), input.data);
    } else {
      LOG(ERROR) << "unsupported feed type " << input.dtype;
    }
  }
}

// Prediction Run
void PredictionRun(PaddlePredictor *predictor,
                   const std::vector<std::vector<PaddleTensor>> &inputs,
                   std::vector<std::vector<PaddleTensor>> *outputs,
                   int num_threads, int tid,
                   const PaddleDType data_type = paddle::PaddleDType::FLOAT32,
                   float *sample_latency = nullptr) {
  int num_times = FLAGS_repeat;
  int iterations = inputs.size();  // process the whole dataset ...
  if (FLAGS_iterations > 0 &&
      FLAGS_iterations < static_cast<int64_t>(inputs.size()))
    iterations =
        FLAGS_iterations;  // ... unless the number of iterations is set
  outputs->resize(iterations);
  LOG(INFO) << "Thread " << tid << ", number of threads " << num_threads
            << ", run " << num_times << " times...";
  int predicted_num = 0;
  if (!FLAGS_zero_copy) {
    for (int i = 0; i < iterations; i++) {
      for (int j = 0; j < num_times; j++) {
        predictor->Run(inputs[i], &(*outputs)[i], FLAGS_batch_size);
      }

      predicted_num += FLAGS_batch_size;
      if (predicted_num % 100 == 0) {
        LOG(INFO) << predicted_num << " samples";
      }
    }
  } else {
    for (int i = 0; i < iterations; i++) {
      ConvertPaddleTensorToZeroCopyTensor(predictor, inputs[i]);
      for (int j = 0; j < num_times; j++) {
        predictor->ZeroCopyRun();
      }

      predicted_num += FLAGS_batch_size;
      if (predicted_num % 100 == 0) {
        LOG(INFO) << predicted_num << " samples";
      }
    }
  }
}

// Create Test Predictor
std::unique_ptr<PaddlePredictor> CreateTestPredictor(
    const PaddlePredictor::Config *config, bool use_analysis = true) {

  const auto *analysis_config =
      reinterpret_cast<const AnalysisConfig *>(config);
  if (use_analysis) {
    return CreatePaddlePredictor<AnalysisConfig>(*analysis_config);
  }
  auto native_config = analysis_config->ToNativeConfig();
  return CreatePaddlePredictor<NativeConfig, paddle::PaddleEngineKind::kNative>(native_config);
}

// Print config
void PrintConfig(const PaddlePredictor::Config *config, bool use_analysis) {
  const auto *analysis_config =
      reinterpret_cast<const AnalysisConfig *>(config);
  if (use_analysis) {
    //LOG(INFO) << *analysis_config;
    return;
  }
  //LOG(INFO) << analysis_config->ToNativeConfig();

}

// Test One Thread
void TestOneThreadPrediction(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    std::vector<std::vector<PaddleTensor>> *outputs, bool use_analysis = true,
    const PaddleDType data_type = paddle::PaddleDType::FLOAT32,
    float *sample_latency = nullptr) {

  auto predictor = CreateTestPredictor(config, use_analysis);
  PredictionRun(predictor.get(), inputs, outputs, 1, 0, data_type,
                sample_latency);
}

// Compare result between two PaddleTensor
void CompareResult(const std::vector<PaddleTensor> &outputs,
                   const std::vector<PaddleTensor> &ref_outputs) {
  EXPECT_GT(outputs.size(), 0UL);
  EXPECT_EQ(outputs.size(), ref_outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    auto &out = outputs[i];
    auto &ref_out = ref_outputs[i];
    size_t size = VecReduceToInt(out.shape);
    size_t ref_size = VecReduceToInt(ref_out.shape);
    EXPECT_GT(size, 0UL);
    EXPECT_EQ(size, ref_size);
    EXPECT_EQ(out.dtype, ref_out.dtype);
    switch (out.dtype) {
      case PaddleDType::INT64: {
        int64_t *pdata = static_cast<int64_t *>(out.data.data());
        int64_t *pdata_ref = static_cast<int64_t *>(ref_out.data.data());
        for (size_t j = 0; j < size; ++j) {
          EXPECT_EQ(pdata_ref[j], pdata[j]);
        }
        break;
      }
      case PaddleDType::FLOAT32: {
        float *pdata = static_cast<float *>(out.data.data());
        float *pdata_ref = static_cast<float *>(ref_out.data.data());
        for (size_t j = 0; j < size; ++j) {
          CHECK_LE(std::abs(pdata_ref[j] - pdata[j]), FLAGS_accuracy);
        }
        break;
      }
      case PaddleDType::INT32: {
        int32_t *pdata = static_cast<int32_t *>(out.data.data());
        int32_t *pdata_ref = static_cast<int32_t *>(ref_out.data.data());
        for (size_t j = 0; j < size; ++j) {
          EXPECT_EQ(pdata_ref[j], pdata[j]);
        }
        break;
      }
    }
  }
}

// Compare Native and Analysis
void CompareNativeAndAnalysis(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs) {
  PrintConfig(config, true);
  std::vector<std::vector<PaddleTensor>> native_outputs, analysis_outputs;
  TestOneThreadPrediction(config, inputs, &native_outputs, false);
  TestOneThreadPrediction(config, inputs, &analysis_outputs, true);
  CompareResult(analysis_outputs.back(), native_outputs.back());
}

void TestMultiThreadPrediction(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    std::vector<std::vector<PaddleTensor>> *outputs, int num_threads,
    bool use_analysis = true) {
  std::vector<std::thread> threads;
  std::vector<std::unique_ptr<PaddlePredictor>> predictors;
  predictors.emplace_back(CreateTestPredictor(config, use_analysis));
  for (int tid = 1; tid < num_threads; tid++) {
    predictors.emplace_back(predictors.front()->Clone());
  }

  for (int tid = 0; tid < num_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      // Each thread should have local inputs and outputs.
      // The inputs of each thread are all the same.
      std::vector<std::vector<PaddleTensor>> outputs_tid;
      auto &predictor = predictors[tid];
      PredictionRun(predictor.get(), inputs, &outputs_tid, num_threads, tid);
    });
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }
}

void TestPrediction(const PaddlePredictor::Config *config,
                    const std::vector<std::vector<PaddleTensor>> &inputs,
                    std::vector<std::vector<PaddleTensor>> *outputs,
                    int num_threads, bool use_analysis = FLAGS_use_analysis) {
  PrintConfig(config, use_analysis);
  if (num_threads == 1) {
    TestOneThreadPrediction(config, inputs, outputs, use_analysis);
  } else {
    TestMultiThreadPrediction(config, inputs, outputs, num_threads,
                              use_analysis);
  }
}

}  // namespace test 
}  // namespace paddle
