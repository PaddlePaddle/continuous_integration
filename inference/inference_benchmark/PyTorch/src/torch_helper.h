#include <map>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <memory>

#include <stdlib.h>
#include <stdio.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <torch/script.h> // One-stop header.
#include <torch/torch.h>


DEFINE_string(model_path, "./mobilenetv1/model",
              "Directory of the infer model file.");
DEFINE_string(params_path, "", "Directory of the infer params file.");
DEFINE_string(model_name, "mobilenetv1", "name of model");
DEFINE_string(model_type, "static", "model generate type");
DEFINE_string(trt_precision, "fp32",
              "tensorrt precision type, choice = ['fp32', 'fp16', 'int8']");
DEFINE_string(image_shape, "3,224,224",
              "can only use for one input model(e.g. image classification)");

DEFINE_bool(use_gpu, false, "use_gpu or not");

DEFINE_int32(thread_num, 1, "num of threads");
DEFINE_int32(batch_size, 1, "batch size");
DEFINE_int32(warmup_times, 10, "warmup times");
DEFINE_int32(repeats, 1000, "repeats times");
DEFINE_int32(cpu_math_library_num_threads, 1, "cpu_math_library_num_threads");
DEFINE_int32(trt_min_subgraph_size, 3, "tensorrt min_subgraph_size");

class Timer {
// Timer, count in ms
  public:
      Timer() {
          reset();
      }
      void start() {
          start_t = std::chrono::high_resolution_clock::now();
      }
      void stop() {
          auto end_t = std::chrono::high_resolution_clock::now();
          typedef std::chrono::microseconds ms;
          auto diff = end_t - start_t;
          ms counter = std::chrono::duration_cast<ms>(diff);
          total_time += counter.count();
      }
      void reset() {
          total_time = 0.;
      }
      double report() {
          return total_time / 1000.0;
      }
  private:
      double total_time;
      std::chrono::high_resolution_clock::time_point start_t;
};

template <typename... Args>
void SummaryConfig(double infer_time, Args... kwargs){
  // TODO: apply Variadic template here

  LOG(INFO) << "----------------------- Model info ----------------------";
  LOG(INFO) << "Model name: " << FLAGS_model_name << ", " \
            << "Model type: " << FLAGS_model_type;
  LOG(INFO) << "----------------------- Data info -----------------------";
  LOG(INFO) << "Batch size: " << FLAGS_batch_size << ", " \
               "Num of samples: " << FLAGS_repeats;
  LOG(INFO) << "----------------------- Conf info -----------------------";
  LOG(INFO) << "device: " << (FLAGS_use_gpu ? "gpu" : "cpu");
  LOG(INFO) << "----------------------- Perf info -----------------------";
  LOG(INFO) << "Average latency(ms): " << infer_time / FLAGS_repeats << ", " \
            << "QPS: " << (FLAGS_repeats * FLAGS_batch_size)/ (infer_time/1000);
}
