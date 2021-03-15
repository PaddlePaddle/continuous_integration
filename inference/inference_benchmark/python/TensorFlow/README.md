# TensorFlow Inference Tests

## load from frozen graph and inference
```shell
python clas_benchmark.py --model_name="resnet50" \
                         --model_path=./resnet_pb/resnet50.pb \
                         --input_node="import/input_tensor:0" \
                         --output_node="softmax_tensor:0"
```

## convert frozen graph .pb to saved_model
```shell
# Notice: input node and output node depends on model's graph
# the follow codes are just an example
python convert_pb2savemodel.py --model_path=./resnet_pb/resnet50.pb \
                               --output_path=./resnet50_model \
                               --input_node="input_tensor:0" \
                               --output_node="softmax_tensor:0"
```

## convert TF fp32 saved_model to TF trt saved_model
```shell
python convert_savemodel2trtgraph.py --model_path=./resnet50_model \
                                     --output_path=./resnet50_model_trt_fp32 \
                                     --trt_precision="fp32"
```

## load from saved model and inference
```shell
python clas_savemodel_benchmark.py --model_path=./resnet50_model_trt_fp32 --use_gpu
```
