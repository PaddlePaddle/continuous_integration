#!/usr/bin/env bash

set -xe

https_proxy=http://172.19.56.199:3128
http_proxy=http://172.19.56.199:3128
#no_proxy=paddlepaddledeps.bj.bcebos.com,github.com

nvidia-docker exec -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e PYTHON_ABI=${PYTHON_ABI} \
    -e CMAKE_BUILD_TYPE=Release \
    -e WITH_GPU=ON \
    -e WITH_DISTRIBUTE=ON \
    -e WITH_MKL=ON \
    -e WITH_NGRAPH=ON \
    -e WITH_AVX=ON \
    -e CUDA_ARCH_NAME=Auto \
    -e WITH_TESTING=OFF \
    -e WITH_COVERAGE=OFF \
    -e LATEST_SUCCESS_BUILD_VCS_NUMBER=${LATEST_SUCCESS_BUILD_VCS_NUMBER} \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    -e no_proxy=${no_proxy} \
    -e BUILD_VCS_NUMBER=${BUILD_VCS_NUMBER} \
    paddle_op_ce /bin/bash ./command_build.sh
