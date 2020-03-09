#!/bin/bash
set -eo pipefail

ROOT=`dirname "$0"`
ROOT=`cd "$ROOT"; pwd`
export CASE_ROOT=$ROOT
export TOOLS_ROOT=$ROOT/tools

if [[ $# -eq 0 ]] ; then
    PADDLE_LIB_PATH=$CASE_ROOT/lib/infer-lib
else
    PADDLE_LIB_PATH=$1
fi

if [[ $# -eq 1 ]] ; then
    CUDA_LIB="/home/work/cuda-9.0/lib64"
else
    PADDLE_LIB_PATH=$1
fi

BUILD=$CASE_ROOT/build
mkdir -p $BUILD
cd $BUILD

cmake $CASE_ROOT/src \
      -DPADDLE_LIB=${PADDLE_LIB_PATH} \
      -DWITH_GPU=ON \
      -DWITH_MKL=ON \
      -DCUDA_LIB=${CUDA_LIB}

make -j4


