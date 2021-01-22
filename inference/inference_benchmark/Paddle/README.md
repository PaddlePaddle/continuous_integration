# Paddle-Inference Benchmark 测试代码


## 快速执行

### 文件简介
`src` 目录为相关测试代码的目录，目前包含以下测试代码。
```shell
demo_helper.h      # 包含各个测试脚本中需要的相同的函数
CMakeLists.txt     # 编译构建文件
bert_benchmark.cc  # bert模型测试脚本，包含TRT dynamic shape
clas_benchmark.cc  # 图像分类模型测试脚本，但单输入模型均可以使用该文件进行测试
eval_clas.cc       # 图像分类模型精度测试脚本，使用的测试数据为 imagenet eval数据集
ptblm_benchmark.cc # ptb-lm模型测试脚本
rcnn_benchmark.cc  # rcnn模型测试脚本
yolo_benchmark.cc  # yolo类型模型测试脚本
```

### 编译
目前该脚本仅支持 Linux 使用，可以修改以下参数快速编译
```shell
DEMO_NAME="all"  # 测试demo名字，可以选择如 clas_benchmark, rcnn_benchmark
WITH_MKL=ON      # 是否使用MKL
WITH_GPU=ON      # 是否使用GPU
USE_TENSORRT=ON  # 是否使用TENSORRT
LIB_DIR="/workspace/paddle_inference_install_dir"
CUDA_LIB="/usr/local/cuda-10.0/lib64"
TENSORRT_ROOT="/usr/local/TensorRT6-cuda10.0-cudnn7"

bash compile.sh ${DEMO_NAME} ${WITH_MKL} ${WITH_GPU} ${USE_TENSORRT} ${LIB_DIR} ${CUDA_LIB} ${TENSORRT_ROOT}
```

### 执行批量测试
`bin` 目录为相关测试的启动脚本，资源监控脚本和日志解析脚本
```shell
# Python脚本
py_mem.py  # 监控预测程序的资源占用情况（GPU+CPU），目前使用于linux x86
py_parse_log.py # 解析预测的日志程序，导出excel表格

# Shell脚本
run_clas_gpu_trt_benchmark.sh 
run_det_gpu_trt_benchmark.sh
run_models_benchmark.sh # 主启动脚本，执行可以直接下载模型并开始预测
```

### 执行单个测试
在完成编译后，如果不想执行批量测试，可以执行单个测试，方法如下，输入变量较多，可以参考 `demo_helper.h` 中声明的进行更改
```shell
./build/clas_benchmark --model_name=${model_name} \
                       --model_path=${model_path} \
                       --params_path=${params_path} \
                       --image_shape=${image_shape} \
                       --batch_size=${batch_size} \
                       --use_gpu=${use_gpu} \
                       --model_type=${MODEL_TYPE} \
                       --trt_precision=${trt_precision} \
                       --trt_min_subgraph_size=${trt_min_subgraph_size} \
                       --use_trt=${use_trt} \
                       --use_mkldnn_=${use_mkldnn_} \
                       --cpu_math_library_num_threads=${cpu_math_library_num_threads}
```


### 输出日志
输出日志样式参考如下，包含C++输出的配置信息和性能信息，Python输出的资源信息
```shell
I0120 10:01:33.694871   516 demo_helper.h:116] ----------------------- Model info ----------------------
I0120 10:01:33.694928   516 demo_helper.h:117] Model name: AlexNet, Model type: static
I0120 10:01:33.694937   516 demo_helper.h:119] ----------------------- Data info -----------------------
I0120 10:01:33.694945   516 demo_helper.h:120] Batch size: 4, Num of samples: 1000
I0120 10:01:33.694953   516 demo_helper.h:122] ----------------------- Conf info -----------------------
I0120 10:01:33.694975   516 demo_helper.h:123] device: gpu, ir_optim: true
I0120 10:01:33.694983   516 demo_helper.h:125] enable_memory_optim: true
I0120 10:01:33.694999   516 demo_helper.h:127] enable_tensorrt: true
I0120 10:01:33.695005   516 demo_helper.h:129] trt_precision: fp16
I0120 10:01:33.695013   516 demo_helper.h:135] ----------------------- Perf info -----------------------
I0120 10:01:33.695017   516 demo_helper.h:136] Average latency(ms): 0.960231, QPS: 4165.66
2021-01-20 10:01:35,749 - __main__ - WARNING - <class 'subprocess.CalledProcessError'>
2021-01-20 10:01:35,750 - __main__ - ERROR - No pid was detected, record process will end
2021-01-20 10:01:35,750 - __main__ - INFO - ----------------------- Res info -----------------------
2021-01-20 10:01:35,750 - __main__ - INFO - process_name: clas_benchmark, cpu rss(MB): 2723.37109375, vms(MB): 10865.59375, shared(MB): 469.6875, dirty(MB): 0.0, cpu_usage(%): 8.277591973244148 
2021-01-20 10:01:35,750 - __main__ - INFO - === gpu info was recorded ===
2021-01-20 10:01:35,750 - __main__ - INFO - gpu_id: 0, total(MB): 16127.625, free(MB): 16127.5625, used(MB): 1916.3125, gpu_utilization_rate(%): 100.0, gpu_mem_utilization_rate(%): 58.0 
```

### 日志导出Excel表格
日志解析脚本为python脚本，使用方法如下，只能解析以上样例中的日志，且保存的日志后缀需要为`.log`
```shell
input_log_path="./log"
output_excel_name="benchmark_model_excel.xlsx"

python bin/py_parse_log.py --log_path=${input_log_path} --output_name=${output_excel_name}
```
