#!/usr/bin/env bash

set -xe

source ~/.bashrc

alias cp=cp
alias rm=rm

cd ${TEST_ROOT_DIR}/bin
rm -rf ../ce_bak/from_model/result/*
python run.py generate_paddle ce_bak/from_model

# run case
cd ${TEST_ROOT_DIR}/script
rm -rf ../ce/${BUILD_VCS_NUMBER}/performance_from_model
mkdir -p ../ce/${BUILD_VCS_NUMBER}/performance_from_model/case
mkdir -p ../ce/${BUILD_VCS_NUMBER}/performance_from_model/log
mkdir -p ../ce/${BUILD_VCS_NUMBER}/performance_from_model/result
cp ../ce_bak/from_model/case/* ../ce/${BUILD_VCS_NUMBER}/performance_from_model/case/
cp ../ce_bak/from_model/result/* ../ce/${BUILD_VCS_NUMBER}/performance_from_model/result/

#tangtianjie adds
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=2
python ce_performance.py ${BUILD_VCS_NUMBER} performance_from_model
