#!/usr/bin/env bash

#set -xe

source ~/.bashrc

alias cp=cp
alias rm=rm

case_pattern=$1

cd ${TEST_ROOT_DIR}/bin
rm -rf ../ce_bak/tf_from_model/*
cp -rf ../ce_bak/from_model/* ../ce_bak/tf_from_model/
rm -rf ../ce_bak/tf_from_model/result/*
python run.py generate_tf ce_bak/tf_from_model
cp -rf ../ce_bak/from_model/result/*.npz ../ce_bak/tf_from_model/result/

# run case
cd ${TEST_ROOT_DIR}/script
rm -rf ../ce/tf_from_model
mkdir ../ce/tf_from_model
mkdir ../ce/tf_from_model/case
mkdir ../ce/tf_from_model/log
mkdir ../ce/tf_from_model/result
cp ../ce_bak/tf_from_model/result/*${case_pattern}.* ../ce/tf_from_model/result/

#tangtianjie adds
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=2
python ce_tf_performance.py tf_from_model
