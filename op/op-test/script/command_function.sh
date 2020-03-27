#!/usr/bin/env bash

set -xe

source ~/.bashrc

alias rm=rm

WORKING_DIR=$1

cd ${TEST_ROOT_DIR}/script
rm -rf ../ce/${BUILD_VCS_NUMBER}
mkdir -p ../ce/${BUILD_VCS_NUMBER}/function/case
mkdir -p ../ce/${BUILD_VCS_NUMBER}/function/log
mkdir -p ../ce/${BUILD_VCS_NUMBER}/function/result
cp ../ce_bak/function/case/* ../ce/${BUILD_VCS_NUMBER}/function/case/

#tangtianjie adds
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=2
python ce_function.py ${BUILD_VCS_NUMBER} function ${WORKING_DIR}/html_output/result.json
