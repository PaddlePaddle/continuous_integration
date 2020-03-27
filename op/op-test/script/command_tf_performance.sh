#!/usr/bin/env bash

set -xe

source ~/.bashrc

alias cp=cp
alias rm=rm

case_pattern=$1

# run case
cd ${TEST_ROOT_DIR}/script
rm -rf ../ce/tf
mkdir ../ce/tf
mkdir ../ce/tf/case
mkdir ../ce/tf/log
mkdir ../ce/tf/result
cp ../ce_bak/tf/result/*${case_pattern}.py ../ce/tf/result/
cp ../ce_bak/tf/result/*${case_pattern}.npz ../ce/tf/result/

#tangtianjie adds
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=2
python ce_tf_performance.py tf
