
#!/bin/sh

ROOT_PATH=$1

export https_proxy=http://172.19.56.199:3128
export http_proxy=http://172.19.56.199:3128

#prepare data
if [ -e data/input ]
then
    echo "args_test_data has already existed"
else
    mkdir -p data/input
    wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/dialogue_domain_classification-dataset-1.0.0.tar.gz
    tar -zxf dialogue_domain_classification-dataset-1.0.0.tar.gz -C ./data/input
fi

#prepare pre_model
if [ -e model ]
then
    echo "model has already existed"
else
    mkdir -p model
    wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/dialogue_domain_classification-model-1.0.0.tar.gz
    tar -zxf dialogue_domain_classification-model-1.0.0.tar.gz -C ./model
fi
