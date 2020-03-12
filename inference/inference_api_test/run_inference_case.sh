#!/bin/bash

sh build_py37_paddel.sh
source init_env.sh

ROOT=`dirname "$0"`
ROOT=`cd "$ROOT/.."; pwd`
CPP_ROOT=${ROOT}/inference_api_test/cpp_api_test
PY_ROOT=${ROOT}/inference_api_test/python_api_test

export PADDLE_ROOT=${ROOT}/inference_api_test/Paddle

cd ${CPP_ROOT}
sh run_cpp.sh

cd ${PY_ROOT}
sh run_case.sh
