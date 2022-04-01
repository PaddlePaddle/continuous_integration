
#!/bin/sh

ROOT_PATH=$1


#prepare data
if [ -e data/input ]
then
    echo "args_test_data has already existed"
else
    mkdir -p data/input
    if [ ! -e dialogue_domain_classification-dataset-1.0.0.tar.gz ]
    then
        echo "download"
        wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/dialogue_domain_classification-dataset-1.0.0.tar.gz
    fi
    tar -zxf dialogue_domain_classification-dataset-1.0.0.tar.gz -C ./data/input
fi

#prepare pre_model
if [ -e model ]
then
    echo "model has already existed"
else
    mkdir -p model
    if [ ! -e dialogue_domain_classification-model-1.0.0.tar.gz ]
    then
        echo "download"
        wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/dialogue_domain_classification-model-1.0.0.tar.gz
    fi
    tar -zxf dialogue_domain_classification-model-1.0.0.tar.gz -C ./model
fi
