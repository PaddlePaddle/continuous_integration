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

LIB_DIR="/workspace/libtorch"
if [ $# -ge 2 ]; then
    LIB_DIR=$2
fi

cmake ../src -DDEMO_NAME=${DEMO_NAME} -DTORCH_LIB=${LIB_DIR}

make -j
