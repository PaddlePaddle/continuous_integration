#!/bin/bash

sh build_paddel.sh
source init_paddle.sh

ROOT=`dirname "$0"`
ROOT=`cd "$ROOT/.."; pwd`
CPP_ROOT=${ROOT}/cpp_api_test
PY_ROOT=${ROOT}/python_api_test

export PADDLE_ROOT=${ROOT}/Paddle

cd CPP_ROOT
sh run_cpp.sh

cd PY_ROOT
sh run_case.sh
