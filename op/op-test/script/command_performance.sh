#!/usr/bin/env bash

set -xe

source ~/.bashrc

alias cp=cp
alias rm=rm

# run case
cd ${TEST_ROOT_DIR}/script
rm -rf ../ce/${BUILD_VCS_NUMBER}/performance
mkdir -p ../ce/${BUILD_VCS_NUMBER}/performance/case
mkdir -p ../ce/${BUILD_VCS_NUMBER}/performance/log
mkdir -p ../ce/${BUILD_VCS_NUMBER}/performance/result
cp ../ce_bak/performance/case/* ../ce/${BUILD_VCS_NUMBER}/performance/case/
cp ../ce_bak/performance/result/*.py ../ce/${BUILD_VCS_NUMBER}/performance/result/
cp ../ce_bak/performance/result/*.npz ../ce/${BUILD_VCS_NUMBER}/performance/result/

#tangtianjie adds
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=2
python ce_performance.py ${BUILD_VCS_NUMBER} performance
