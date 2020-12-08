#!/bin/bash
set -ex

mkdir -p build
cd build

# same with the class_benchmark_demo.cc

# DEMO_NAME=clas_benchmark
DEMO_NAME=all
if [ $# -ge 1 ]; then
    DEMO_NAME=$1
fi

WITH_MKL=ON
if [ $# -ge 2 ]; then
    WITH_MKL=$2
fi

WITH_GPU=ON
if [ $# -ge 3 ]; then
    WITH_GPU=$3
fi

USE_TENSORRT=ON
if [ $# -ge 4 ]; then
    USE_TENSORRT=$4
fi

LIB_DIR="/workspace/paddle_inference_install_dir"
if [ $# -ge 5 ]; then
    LIB_DIR=$5
fi

CUDA_LIB="/usr/local/cuda-10.0/lib64"
if [ $# -ge 6 ]; then
    CUDA_LIB=$6
fi

TENSORRT_ROOT="/usr/local/TensorRT6-cuda10.0-cudnn7"
if [ $# -ge 7 ]; then
    TENSORRT_ROOT=$7
fi

cmake ../src -DPADDLE_LIB=${LIB_DIR} \
             -DWITH_MKL=${WITH_MKL} \
             -DDEMO_NAME=${DEMO_NAME} \
             -DWITH_GPU=${WITH_GPU} \
             -DWITH_STATIC_LIB=OFF \
             -DUSE_TENSORRT=${USE_TENSORRT} \
             -DCUDNN_LIB=${CUDNN_LIB} \
             -DCUDA_LIB=${CUDA_LIB} \
             -DTENSORRT_ROOT=${TENSORRT_ROOT}

make -j
