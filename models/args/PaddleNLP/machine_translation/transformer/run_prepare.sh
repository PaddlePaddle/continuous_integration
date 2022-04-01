#!/bin/bash

ROOT_PATH=$1

if [ -e data ]
then
   mv data data.bak
fi
if [ ! -e data.tgz ]
then
    wget https://sys-p0.bj.bcebos.com/models/PaddleNLP/machine_translation/transformer/data.tgz --no-check-certificate
fi
tar -zxf data.tgz

#prepare data
if [ -e args_test_data ]
then
    echo "args_test_data has already existed"
else
    ln -s data args_test_data
fi

rm -rf args_test_inference_model
mkdir args_test_inference_model

rm -rf args_test_finetuned
mkdir args_test_finetuned

#prepare pre_model

if [ -e args_test_model ]
then
    echo "args_test_model has already existed"
else
    ln -s data/args_test_model args_test_model
fi
