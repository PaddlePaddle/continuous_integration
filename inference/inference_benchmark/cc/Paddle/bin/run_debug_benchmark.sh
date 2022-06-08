#!/usr/bin/env bash
set -eo pipefail

ROOT=`dirname "$0"`
ROOT=`cd "$ROOT/.."; pwd`
export OUTPUT=$ROOT/output
export OUTPUT_BIN=$ROOT/build
export DATA_ROOT=$ROOT/Data
export CASE_ROOT=$ROOT/bin
export LOG_ROOT=$ROOT/log
export UTILS_ROOT=$ROOT/utils
export gpu_type=`nvidia-smi -q | grep "Product Name" | head -n 1 | awk '{print $NF}'`
source $ROOT/bin/run_clas_mkl_benchmark.sh
#source $ROOT/bin/run_det_mkl_benchmark.sh
# test model type
model_type="static"
if [ $# -ge 1 ]; then
    model_type=$1
fi
export MODEL_TYPE=${model_type}

# test run-time device
device_type="gpu"
if [ $# -ge 2 ]; then
    device_type=$2
fi

mkdir -p $DATA_ROOT
cd $DATA_ROOT
if [ ! -f PaddleClas/infer_static/AlexNet/__model__ ]; then
    echo "==== Download PaddleClas data and models ===="
    wget --no-proxy -q https://sys-p0.bj.bcebos.com/Paddle-UnitTest-Model/PaddleClas.tgz --no-check-certificate
    tar -zxf PaddleClas.tgz
fi

cd -

mkdir -p $LOG_ROOT

echo "==== run ${MODEL_TYPE} model benchmark ===="

if [ "${MODEL_TYPE}" == "static" ]; then
    if [ "${device_type}" == "gpu" ]; then
        bash $CASE_ROOT/run_debug_clas_gpu_trt_benchmark.sh "${DATA_ROOT}/PaddleClas/infer_static"
    fi
fi
