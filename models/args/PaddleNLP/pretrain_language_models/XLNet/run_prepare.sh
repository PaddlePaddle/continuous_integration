#!/bin/bash

#prepare data
if [ -e args_test_data ]
then
    echo "args_test_data has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/PaddleLARK/XLNet/data args_test_data
fi


#prepare pre_model
if [ -e xlnet_cased_L-12_H-768_A-12 ]
then
    echo "xlnet_cased_L-12_H-768_A-12 has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/PaddleLARK/XLNet/xlnet_cased_L-12_H-768_A-12 xlnet_cased_L-12_H-768_A-12
fi

