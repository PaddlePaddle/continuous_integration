#!/usr/bin/env bash

set -eo pipefail

ROOT=`dirname "$0"`
ROOT=`cd "$ROOT/.."; pwd`
export OUTPUT=$ROOT/output
export OUTPUT_BIN=$ROOT/build
export DATA_ROOT=$ROOT/Data
export CASE_ROOT=$ROOT/bin
export LOG_ROOT=$ROOT/log
export gpu_type=`nvidia-smi -q | grep "Product Name" | head -n 1 | awk '{print $NF}'`

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

echo "==== run ${MODEL_TYPE} model pressure tests ===="
echo "==== run tests on ${device_type} ==== "

if [ "${MODEL_TYPE}" == "static" ]; then
    if [ "${device_type}" == "gpu" ]; then
        echo "==== start ${device_type} tests  ==== "
        model_name="ResNet50"
        model_path=${DATA_ROOT}/PaddleClas/infer_static/${model_name}/__model__
        params_path=${DATA_ROOT}/PaddleClas/infer_static/${model_name}/params
        image_shape="3,224,224"
        batch_size=6
        use_gpu=True
        trt_precision="fp32"
        trt_min_subgraph_size="3"
        use_trt=True
        repeats=100000

        $OUTPUT_BIN/clas_benchmark --model_name=${model_name} \
                                   --model_path=${model_path} \
                                   --params_path=${params_path} \
                                   --image_shape=${image_shape} \
                                   --batch_size=${batch_size} \
                                   --use_gpu=${use_gpu} \
                                   --model_type=${MODEL_TYPE} \
                                   --trt_precision=${trt_precision} \
                                   --trt_min_subgraph_size=${trt_min_subgraph_size} \
                                   --use_trt=${use_trt} \
                                   --repeats=${repeats}
        echo "==== finish ${device_type} tests  ==== "
    elif [ "${device_type}" == "cpu" ]; then
        echo "==== start ${device_type} tests  ==== "
        model_name="ResNet50"
        model_path=${DATA_ROOT}/PaddleClas/infer_static/${model_name}/__model__
        params_path=${DATA_ROOT}/PaddleClas/infer_static/${model_name}/params
        image_shape="3,224,224"
        batch_size=4
        use_gpu=False

        $OUTPUT_BIN/clas_benchmark --model_name=${model_name} \
                                   --model_path=${model_path} \
                                   --params_path=${params_path} \
                                   --image_shape=${image_shape} \
                                   --batch_size=${batch_size} \
                                   --use_gpu=${use_gpu} \
                                   --model_type=${MODEL_TYPE} \
                                   --repeats=${repeats}
        echo "==== finish ${device_type} tests  ==== "
    fi
elif [ "${MODEL_TYPE}" == "dy2static" ]; then
    echo "current not support ${device_type}"
fi




