#!/bin/bash

ROOT_PATH=$1

#prepare data
if [ -e args_test_data ]
then
    echo "args_test_data has already existed"
else
    #ln -s /ssd3/models_test/models_args/PaddleNLP/PaddleMT/transformer/data args_test_data
    ln -s ${ROOT_PATH}/data/PaddleNLP/machine_translation/transformer/data args_test_data
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
    #ln -s /ssd3/models_test/models_args/PaddleNLP/PaddleMT/transformer/args_test_model args_test_model
    ln -s ${ROOT_PATH}/data/PaddleNLP/machine_translation/transformer/args_test_model args_test_model
fi
