#!/bin/bash
#REPO=${REPO}
BRANCH=${BRANCH:-develop}
#AGILE_PULL_ID=$3
#AGILE_REVISION=$4
ROOT_PATH=${ROOT_PATH:-/home/work/tipc/}
paddle_package=${paddle_package:-xly-devops/PR/Paddle}

work_dir=${ROOT_PATH}/${REPO}
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
#git clone --depth=100 https://github.com/PaddlePaddle/continuous_integration.git;
git clone --depth=100 https://github.com/zhengya01/continuous_integration.git -b v1;
if [[ $REPO == PaddleClas ]]; then
  #git clone https://github.com/huangjun12/PaddleVideo.git -b benchmark-0728 
  #git clone --depth=2 https://github.com/PaddlePaddle/${REPO}.git -b ${BRANCH};
  #git clone https://github.com/zhengya01/PaddleOCR.git -b tipc_log
  git clone https://github.com/TingquanGao/PaddleClas.git -b dev/tipc_fix_log_path
else
  git clone --depth=2 https://github.com/PaddlePaddle/${REPO}.git -b ${BRANCH};
fi



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
