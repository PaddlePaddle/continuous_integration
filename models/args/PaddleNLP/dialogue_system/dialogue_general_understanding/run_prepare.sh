#!/bin/bash

ROOT_PATH=$1

if [ -e data ]
then
   mv data data.bak
fi
if [ ! -e data.tgz ]
then
    wget https://sys-p0.bj.bcebos.com/models/PaddleNLP/dialogue_system/dialogue_general_understanding/data.tgz --no-check-certificate
fi
tar -zxf data.tgz

#prepare data
if [ -e args_test_data ]
then
    echo "args_test_data has already existed"
else
    #ln -s /ssd3/models_test/models_args/PaddleNLP/PaddleDialogue/dialogue_general_understanding/data args_test_data
    ln -s data args_test_data
fi

rm -rf args_test_save_model
mkdir args_test_save_model

rm -rf args_test_inference_model
mkdir args_test_inference_model

#prepare pre_model
