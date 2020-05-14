#!/bin/sh

pip install numpy

#prepare data
if [ -e args_test_senta_data ]
then
    echo "args_test_senta_data has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/sentiment_classification/senta_data args_test_senta_data
fi
if [ -e args_test_senta_data_1 ]
then
    echo "args_test_senta_data_1 has already existed"
else
    ln -s /ssd3/models_test/models_args/PaddleNLP/sentiment_classification/senta_data args_test_senta_data_1
fi

#prepare model
if [ -e args_test_senta_model ]
then
    echo "args_test_senta_model has already existed"
else
    #wget https://baidu-nlp.bj.bcebos.com/sentiment_classification-1.0.0.tar.gz --no-check-certificate
    #tar -zxf sentiment_classification-1.0.0.tar.gz 
    #/bin/rm sentiment_classification-1.0.0.tar.gz
    #mv senta_model args_test_senta_model
    ln -s /ssd3/models_test/models_args/PaddleNLP/sentiment_classification/senta_model args_test_senta_model
fi
