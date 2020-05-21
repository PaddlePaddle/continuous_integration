#!/bin/bash

#prepare data
if [ -e args_test_data ]
then
    echo "args_test_data has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/PaddleDialogue/dialogue_general_understanding/data args_test_data
fi

rm -rf args_test_save_model
mkdir args_test_save_model

rm -rf args_test_inference_model
mkdir args_test_inference_model

#prepare pre_model

