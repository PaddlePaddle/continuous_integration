# Paddle-Inference CPU Benchmark

## quick start

### compile inference demo
shell scripts only support linux platfrom right now.
windows bat scripts will be added later
```shell
DEMO_NAME="all"   # name of test demo, it can be clas_benchmark, rcnn_benchmark
WITH_MKL=ON       # whether to use MKL
WITH_GPU=OFF      # whether to use GPU
USE_TENSORRT=OFF  # whether to use TENSORRT
LIB_DIR="/workspace/paddle_inference_install_dir" # path of cpp inference lib

bash compile.sh ${DEMO_NAME} ${WITH_MKL} ${WITH_GPU} ${USE_TENSORRT} ${LIB_DIR}
```

### run batch test scripts
`bin` directory has starting scripts, resources monitor scripts and log parser.
```shell
# Python scripts
py_mem.py  # monitor inference bin（GPU+CPU），only support linux x86 currently
py_parse_log.py # parse inference logs and covert to excel

# Shell scripts
run_models_benchmark.sh # main scripts, execute will download models if it not exist
run_clas_mkl_benchmark.sh
run_det_mkl_benchmark.sh
```

### whole process
```shell
# 1. download inference-lib or compile from source codes

# 2. download test codes
git clone https://github.com/PaddlePaddle/continuous_integration.git

# 3. compile test codes
cd continuous_integration/inference/inference_benchmark/cc/Paddle
bash compile.sh all ON OFF OFF /path/of/inference-lib

# 4. prepare python third-party lib
python3.7 -m pip install py3nvml;  # monitor gpu
python3.7 -m pip install cup;      # monitor cpu
python3.7 -m pip install pandas;   # process data
python3.7 -m pip install openpyxl; # process data to excel


# 5. start benchmark tests
bash bin/run_models_benchmark.sh "static" "cpu" "1" "1"

# 6. convert data to excel
python3.7 bin/py_parse_log.py --log_path=./log --output_name=benchmark_mkl_excel.xlsx
```