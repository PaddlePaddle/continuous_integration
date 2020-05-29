#pragma once

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

namespace paddle {
namespace test {
// Parse tensor from string
template <typename T>
bool ParseTensor(const std::string &field, paddle::PaddleTensor *tensor) {
    std::vector<std::string> data;
    Split(field, ':', &data);
    if (data.size() < 2) {
        return false;
        }
    std::string shape_str = data[0];
    std::vector<int> shape;
    Split(shape_str, ' ', &shape);
    std::string mat_str = data[1];
    std::vector<T> mat;
    Split(mat_str, ' ', &mat);
    tensor->shape = shape;
    auto size =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
        sizeof(T);
    tensor->data.Resize(size);
    std::copy(mat.begin(), mat.end(), static_cast<T *>(tensor->data.data()));
    tensor->dtype = GetPaddleDType<T>();
    return true;
}

// Parse input tensors from string
bool ParseLine(const std::string &line,
               std::vector<paddle::PaddleTensor> *tensors) {
    std::vector<std::string> fields;
    Split(line, ';', &fields);
    if (fields.size() < 5) return false;
    tensors->clear();
    tensors->reserve(5);
    int i = 0;
    // src_id
    paddle::PaddleTensor src_id;
    ParseTensor<int64_t>(fields[i++], &src_id);
    tensors->push_back(src_id);
    // pos_id
    paddle::PaddleTensor pos_id;
    ParseTensor<int64_t>(fields[i++], &pos_id);
    tensors->push_back(pos_id);
    // segment_id
    paddle::PaddleTensor segment_id;
    ParseTensor<int64_t>(fields[i++], &segment_id);
    tensors->push_back(segment_id);
    // self_attention_bias
    paddle::PaddleTensor self_attention_bias;
    ParseTensor<float>(fields[i++], &self_attention_bias);
    tensors->push_back(self_attention_bias);
    // next_segment_index
    paddle::PaddleTensor next_segment_index;
    ParseTensor<int64_t>(fields[i++], &next_segment_index);
    tensors->push_back(next_segment_index);
    return true;
}

// Load input data
bool LoadInputData(std::vector<std::vector<paddle::PaddleTensor>> *inputs) {
    if (FLAGS_infer_data.empty()) {
        LOG(ERROR) << "please set input data path";
    return false;
    }
    std::ifstream fin(FLAGS_infer_data);
    std::string line;
    int sample = 0;
    while (std::getline(fin, line)) {
        std::vector<paddle::PaddleTensor> feed_data;
        ParseLine(line, &feed_data);
        inputs->push_back(std::move(feed_data));
        sample++;
        if (!FLAGS_test_all_data && sample == FLAGS_batch_size) break;
    }
    LOG(INFO) << "number of samples: " << sample;
    return true;
}
} //namespace test 
} //namespace paddle