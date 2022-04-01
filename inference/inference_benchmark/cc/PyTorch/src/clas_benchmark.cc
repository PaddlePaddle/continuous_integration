#include "./torch_helper.h"

namespace torch_infer {
  torch::Device LoadModel(torch::jit::script::Module *module){
    torch::Device device = torch::Device(FLAGS_use_gpu ? "cuda:0" : "cpu");
    module->to(device);
    return device;
  }

  double Inference(torch::jit::script::Module *module, 
                   torch::Device device){
    std::cout << "create input tensor..." << std::endl;
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({FLAGS_batch_size, 3, 224,224}).to(device));
    
    // warmup
    std::cout << "warm up..." << std::endl;
    for(int i=0; i < FLAGS_warmup_times; ++i){
      at::Tensor output = module->forward(inputs).toTensor();
    }
    
    // main process
    std::cout << "start to inference..." << std::endl;
    Timer pred_timer;  // init prediction timer
    pred_timer.start();  // start timer
    for(int i=0; i < FLAGS_repeats; ++i){
      at::Tensor output = module->forward(inputs).toTensor();
    }
    pred_timer.stop();  // stop timer
    return pred_timer.report();
}

int RunDemo() {
  torch::jit::script::Module module = torch::jit::load(FLAGS_model_path);
  torch::Device device = LoadModel(&module);
  auto total_time = Inference(&module, device);
  SummaryConfig(total_time);
  return 0;
}

}  // namespace torch_infer


int main(int argc, char**argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  torch_infer::RunDemo();
  return 0;
}


