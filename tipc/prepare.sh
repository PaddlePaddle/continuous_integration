#!/bin/bash
REPO=$1
BRANCH=${2:-develop}
AGILE_PULL_ID=$3
AGILE_REVISION=$4

work_dir=/home/work/tipc/${REPO}
mkdir -p ${work_dir}
rm -rf ${work_dir}/*
cd ${work_dir}

unset http_proxy
unset https_proxy

# Git clone
if [ -d Paddle ]; then rm -rf Paddle; fi 
git config --global user.name "PaddleCI"
git config --global user.email "paddle_ci@example.com"
git clone --depth=200 https://github.com/PaddlePaddle/Paddle.git
cd Paddle
#git fetch origin pull/${AGILE_PULL_ID}/head
#git checkout -b test FETCH_HEAD

# download test model repo
git clone --depth=100 https://github.com/LDOUBLEV/AutoLog;
git clone --depth=100 https://github.com/PaddlePaddle/continuous_integration.git;
git clone --depth=2 https://github.com/PaddlePaddle/${REPO}.git -b ${BRANCH};

cd ${work_dir}
tar -zcf ${REPO}.tar.gz Paddle
file_tgz=${REPO}.tar.gz

# Push BOS
# pip install pycrypto
push_dir=/home
push_file=${push_dir}/bce-python-sdk-0.8.27/BosClient.py
if [ ! -f ${push_file} ];then
    set +x
    wget -q --no-proxy -O /home/bce_whl.tar.gz  https://paddle-docker-tar.bj.bcebos.com/home/bce_whl.tar.gz --no-check-certificate
    set -x
    tar xf /home/bce_whl.tar.gz -C ${push_dir}
fi

cd ${work_dir}
python ${push_file}  ${file_tgz}  ${paddle_package}/fullchain_ce_test/${AGILE_PULL_ID}/${AGILE_REVISION}
