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

WITH_GPU=ON
if [ $# -ge 2 ]; then
    WITH_GPU=$2
fi

WITH_MKL=OFF
if [ $# -ge 3 ]; then
    WITH_MKL=$3
fi

USE_TENSORRT=ON
if [ $# -ge 4 ]; then
    USE_TENSORRT=$4
fi

TENSORRT_ROOT="/usr/lib/aarch64-linux-gnu/"
if [ $# -ge 5 ]; then
    TENSORRT_ROOT=$5
fi

export CUDA_LIB="/usr/local/cuda/lib64"

BUILD=$CASE_ROOT/build
mkdir -p $BUILD
cd $BUILD

cmake $CASE_ROOT/src \
      -DPADDLE_LIB=${PADDLE_LIB_PATH} \
      -DWITH_GPU=${WITH_GPU} \
      -DCUDA_LIB=${CUDA_LIB} \
      -DUSE_TENSORRT=${USE_TENSORRT} \
      -DTENSORRT_INCLUDE_DIR="${TENSORRT_ROOT}/include" \
      -DTENSORRT_LIB_DIR="${TENSORRT_ROOT}"

make -j4