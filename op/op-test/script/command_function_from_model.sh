#!/usr/bin/env bash

set -xe

source ~/.bashrc

alias rm=rm
alias cp=cp

WORKING_DIR=$1

cd ${TEST_ROOT_DIR}/bin
rm -rf ../ce/function_from_model/*
cp -rf ../ce_bak/from_model/* ../ce/function_from_model/
rm -rf ../ce/function_from_model/result/*
python run.py generate_function ce/function_from_model

cd ${TEST_ROOT_DIR}/script
rm -rf ../ce/${BUILD_VCS_NUMBER}/function_from_model
mkdir -p ../ce/${BUILD_VCS_NUMBER}/function_from_model/case
mkdir -p ../ce/${BUILD_VCS_NUMBER}/function_from_model/log
mkdir -p ../ce/${BUILD_VCS_NUMBER}/function_from_model/result
cp ../ce_bak/from_model/case/* ../ce/${BUILD_VCS_NUMBER}/function_from_model/case/

#tangtianjie adds
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=2
python ce_function.py ${BUILD_VCS_NUMBER} function_from_model ${WORKING_DIR}/html_output/result_from_model.json
