#!/bin/bash

#prepare data
if [ -e args_test_data_1 ]
then
    echo "args_test_data_1 has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/language_model/data args_test_data_1 
fi
if [ -e args_test_data_2 ]
then
    echo "args_test_data_2 has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/language_model/data args_test_data_2 
fi

if [ -e models_1 ]
then
    echo "models_1 has already existed"
else 
    mkdir models_1
    cd models_1
    mkdir 0 1 2 3 4 5
fi

#prepare pre_model

