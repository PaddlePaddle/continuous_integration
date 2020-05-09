#!/bin/bash

ROOT_PATH=$1

#prepare data
if [ -e args_test_data ]
then
    echo "args_test_data has already existed"
    rm args_test_data
fi
ln -s ${ROOT_PATH}/data/PaddleNLP/dialogue_system/auto_dialogue_evaluation/data args_test_data

pwd
ls -l

rm -rf args_test_inference_model
mkdir args_test_inference_model

rm -rf args_test_finetuned
mkdir args_test_finetuned
