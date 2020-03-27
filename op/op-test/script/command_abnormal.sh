#!/usr/bin/env bash

set -xe

source ~/.bashrc

alias rm=rm

WORKING_DIR=$1

cd ${TEST_ROOT_DIR}/script
#rm ../ce/abnormal/case/*
#rm ../ce/abnormal/log/*
#rm ../ce/abnormal/result/*
python generate_case.py

#tangtianjie adds
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=2
python ce_function.py "" abnormal ${WORKING_DIR}/html_output/abnormal_result.json
