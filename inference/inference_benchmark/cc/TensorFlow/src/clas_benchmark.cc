#include "./tf_helper.h"

namespace tf_infer {

typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;


tensorflow::Status LoadModel(tensorflow::Session *sess, std::string graph_fn,
                             std::string checkpoint_fn = "") {
  tensorflow::Status status;

  // Read in the protobuf graph we exported
  tensorflow::GraphDef graph_def;
  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
  if (status != tensorflow::Status::OK()) return status;

  // create the graph
  // tensorflow::graph::SetDefaultDevice("/gpu:3", const_cast<tensorflow::GraphDef*>(&graph_def.graph_def()));
  //tensorflow::graph::SetDefaultDevice("/gpu:3", const_cast<tensorflow::GraphDef*>(&graph_def));
  // tensorflow::graph::SetDefaultDevice("/gpu:3", &graph_def);
  status = sess->Create(graph_def);
  if (status != tensorflow::Status::OK()) return status;

  return tensorflow::Status::OK();
}


double Inference(tensorflow::Session *sess){
  // prepare inputs
  tensorflow::TensorShape data_shape({1, 224, 224, 3});
  tensorflow::Tensor data(tensorflow::DT_FLOAT, data_shape);

  // same as in python file
  const std::string input_name = "input:0";
  const std::string output_name = "MobilenetV2/Predictions/Reshape_1:0";

  tensor_dict feed_dict = {
      {input_name, data},
  };

  std::vector<tensorflow::Tensor> outputs;

  // warmup
  for(int i=0; i < FLAGS_warmup_times; ++i){
    TF_CHECK_OK(sess->Run(feed_dict, {output_name}, {}, &outputs));
  }

  Timer pred_timer;  // init prediction timer
  pred_timer.start();  // start timer
  for(int i=0; i < FLAGS_repeats; ++i){
    TF_CHECK_OK(sess->Run(feed_dict, {output_name}, {}, &outputs));
  }
  pred_timer.stop();  // stop timer

  return pred_timer.report();
}


int RunDemo() {
  const std::string graph_fn = FLAGS_model_path;

  // TF_SessionOptions* option = TF_NewSessionOptions();
  // TF_EnableXLACompilation(option, true);

  // prepare session
  tensorflow::Session *sess;
  tensorflow::SessionOptions options;
  TF_CHECK_OK(tensorflow::NewSession(options, &sess));

  tensorflow::ConfigProto* config = &options.config;
  // // disabled GPU entirely
  // (*config->mutable_device_count())["GPU"] = 0;
  // // place nodes somewhere
  // config->set_allow_soft_placement(true);
  
  TF_CHECK_OK(LoadModel(sess, graph_fn));

  auto total_time = Inference(sess);

  SummaryConfig(total_time);

  return 0;
}

}  // namespace tf_infer

int main(int argc, char**argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  tf_infer::RunDemo();
  return 0;
}
