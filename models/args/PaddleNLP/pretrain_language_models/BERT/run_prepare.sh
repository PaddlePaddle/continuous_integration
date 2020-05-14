#!/bin/bash

#prepare data
if [ -e args_test_data ]
then
    echo "args_test_data has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/PaddleLARK/BERT/data args_test_data
fi


#prepare pre_model
if [ -e args_test_model ]
then
    echo "args_test_model has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/PaddleLARK/BERT/pretrain_model args_test_model
fi


rm -rf args_test_output
mkdir args_test_output
