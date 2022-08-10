#!/bin/bash

model_name=$1
log_path=$2
CHAIN=$3

# prepare SDK
push_dir=$PWD
push_file=${push_dir}/bce-python-sdk-0.8.27/BosClient.py
cp /workspace/bce_whl.tar.gz ./
if [ ! -f ${push_dir}/bce_whl.tar.gz ];then
    echo "BOS SDK pull failed"
    exit 1
fi
if [ ! -f ${push_file} ];then
    tar xf $PWD/bce_whl.tar.gz -C ${push_dir}
fi

# get model info
time_stamp=`date +%Y_%m_%d_%H_%m_%s`
cd "/workspace/${REPO}"
repo_commit=`git rev-parse HEAD`
cd -
cd /workspace
# paddle_commit 需通过安装查询 todo
paddle_commit=`git rev-parse HEAD`
cd -

# copy log file
log_file=`echo "${log_path}" | awk -F '/' '{print $NF}'`
upload_file="${time_stamp}${REPO}^${CHAIN}^${model_name}^${paddle_commit}^${repo_commit}^${log_file}"
cp ${log_path} ${upload_file}
python2 ${push_file} ${upload_file} paddle-qa/fullchain_test/logs/
exit_code=$?
model_url="https://paddle-qa.bj.bcebos.com/fullchain_test/logs/${upload_file}"
if [[ exit_code -eq 0 ]]
then
    echo $model_url
fi
rm -rf ${upload_file}
exit $exit_code

