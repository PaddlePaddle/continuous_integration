#!/bin/bash

REPO=$1
model_path_in_docker=$2
# example
# model_path_in_docker="/workspace/PaddleNLP/tests/infer_model,transformer:/workspace/PaddleNLP/tests/test_tipc/bigru_crf/infer_model,bigru_crf"

if [[ ${model_path_in_docker} == "" ]]; then
    echo 'model_path_in_docker not passed'
    exit 0
fi

# Push BOS
python2 -m pip install --retries 10 pycrypto -i https://mirror.baidu.com/pypi/simple
push_dir=$PWD
push_file=${push_dir}/bce-python-sdk-0.8.27/BosClient.py
if [ ! -f ${push_file} ];then
    set +x
    wget -q --no-proxy -O $PWD/bce_whl.tar.gz  https://paddle-docker-tar.bj.bcebos.com/home/bce_whl.tar.gz --no-check-certificate
    set -x
    tar xf $PWD/bce_whl.tar.gz -C ${push_dir}
fi
time_stamp=`date +%Y_%m_%d`
cd "/workspace/${REPO}"
repo_commit=`git rev-parse HEAD`
cd -
cd /workspace
paddle_commit=`git rev-parse HEAD`
cd -

IFS=":"
model_array=(${model_path_in_docker})

for single_model in ${model_array[@]}
do
    echo ${single_model}
    IFS=","
    array=(${single_model})
    model_path=${array[0]}
    model_tar_name="${time_stamp}^${REPO}^${array[1]}^${paddle_commit}^${repo_commit}.tgz"
    echo ${model_path}
    echo ${model_name}
    tar -zcvf ${model_tar_name} ${model_path}
    python2 ${push_file} ${model_tar_name} paddle-qa/fullchain_ce_test
done
