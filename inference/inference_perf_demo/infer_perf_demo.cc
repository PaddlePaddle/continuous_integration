#include <numeric>
#include <thread>
#include <iostream>
#include <memory>
#include <chrono>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"

DEFINE_string(model_path, "./mobilenetv1", "Directory of the inference model.");
DEFINE_string(model_name, "mobilenetv1", "name of model");
DEFINE_string(model_type, "static", "model generate type");

DEFINE_bool(use_gpu, false, "use_gpu or not");
DEFINE_bool(use_trt, false, "use trt or not");
DEFINE_bool(use_mkldnn, false, "use mkldnn or not");

DEFINE_int32(thread_num, 1, "thread_num");
DEFINE_int32(batch_size, 1, "batch_size");
DEFINE_int32(warmup_times, 10, "warmup");
DEFINE_int32(repeats, 1000, "repeats");


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


namespace paddle_infer {

void PrepareConfig(Config *config) {
  // prepare Paddle-Inference Config
  config->SetModel(FLAGS_model_path + "/__model__",
                   FLAGS_model_path + "/__params__");
  if (FLAGS_use_gpu){
    config->EnableUseGpu(100, 0);
    if (FLAGS_use_trt) {
      config->EnableTensorRtEngine();
    }
  }
  else {
    config->DisableGpu();
    if (FLAGS_use_mkldnn) {
      config->EnableMKLDNN();
      LOG(INFO) << "mkldnn enabled";
    }
  }
}

double Inference(Predictor* pred, int tid) {
  int channels = 3;
  int height = 224;
  int width = 224;
  int batch_size = FLAGS_batch_size;
  int input_num = channels * height * width * batch_size;

  // prepare inputs
  std::vector<float> in_data(input_num);
  for (int i=0; i < input_num; ++i) {
    in_data[i] = i % 10 * 0.1;
  }

  // set inputs
  auto in_names = pred->GetInputNames();
  auto input_t = pred->GetInputHandle(in_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->CopyFromCpu(in_data.data());

  // warm-up
  for (size_t i = 0; i < FLAGS_warmup_times; ++i) {
    pred->Run();
  }

  Timer pred_timer; // init prediction timer
  int out_num = 0;
  std::vector<float> out_data;

  // main prediction process
  pred_timer.start(); // start timer
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    pred->Run();
    auto out_names = pred->GetOutputNames();
    auto output_t = pred->GetOutputHandle(out_names[0]);

    std::vector<int> output_shape = output_t->shape();
    // retrive date to output vector
    out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    out_data.resize(out_num);
    output_t->CopyToCpu(out_data.data());
  }
  pred_timer.stop(); // stop timer

  return pred_timer.report();
}

void RunDemo() {
  Config config;
  PrepareConfig(&config);

  services::PredictorPool pred_pool(config, FLAGS_thread_num);

  auto total_time = Inference(pred_pool.Retrive(0), 0);

  LOG(INFO) << "----------------------- Model info ----------------------";
  LOG(INFO) << "Model name: " << FLAGS_model_name << ", Model type: " << FLAGS_model_type;
  LOG(INFO) << "----------------------- Data info -----------------------";
  LOG(INFO) << "Batch size: " << FLAGS_batch_size << ", Num of samples: " << FLAGS_repeats;
  LOG(INFO) << "----------------------- Conf info -----------------------";
  LOG(INFO) << "device: " << (config.use_gpu() ? "gpu" : "cpu") << ", thread num: " << 2 << ", ir_optim: " << (config.ir_optim() ? "true" : "false");
  if (FLAGS_use_gpu){
    if (FLAGS_use_trt) {
      LOG(INFO) << "enable_tensorrt: " << (config.tensorrt_engine_enabled() ? "true" : "false");
    }
  }
  else {
    LOG(INFO) << "cpu_math_library_num_threads: " << config.cpu_math_library_num_threads();
    if (FLAGS_use_mkldnn) {
      LOG(INFO) << "enable_mkldnn: " << (config.mkldnn_enabled() ? "true" : "false");
    }
  }
  LOG(INFO) << "----------------------- Perf info -----------------------";
  LOG(INFO) << "Average latency(ms): " << total_time / static_cast<float>(FLAGS_repeats) << ", QPS: " << (FLAGS_repeats * FLAGS_batch_size)/ (total_time/1000) ;
}

} // namespace paddle_infer


int main(int argc, char**argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  paddle_infer::RunDemo();
  return 0;
}

