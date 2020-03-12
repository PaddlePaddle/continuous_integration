#!/bin/bash

echo ${PADDLE_ROOT}
export CUDA_LIB="/usr/local/cuda-10.0/targets/x86_64-linux/lib"

sh build.sh "${PADDLE_LIB}/build/fluid_inference_install_dir" ${CUDA_LIB}
if [ $? -ne 0 ];then
    echo -e "\033[33m build test case failed! \033[0m";exit;
fi

sh bin/run-case.sh

