#!/bin/bash

ROOT_PATH=$1

#prepare data

if [ -e data ]
then
   mv data data.bak
fi
if [ ! -e data.tgz ]
then
    wget https://sys-p0.bj.bcebos.com/models/PaddleNLP/language_model/data.tgz --no-check-certificate
fi
tar -zxf data.tgz


if [ -e args_test_data_1 ]
then
    echo "args_test_data_1 has already existed"
else
    ln -s data args_test_data_1 
fi
if [ -e args_test_data_2 ]
then
    echo "args_test_data_2 has already existed"
else
    ln -s data args_test_data_2 
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

