#!/bin/bash

project_path=$(cd "$(dirname "$0")";pwd)

if [ -d "Data" ];then rm -rf Data
fi
mkdir -p ./Data
cd ./Data
wget -q https://sys-p0.bj.bcebos.com/inference/python-infer.tgz --no-check-certificate
tar -xvf python-infer.tgz
wget -q https://sys-p0.bj.bcebos.com/inference/python-model-infer.tgz --no-check-certificate
tar -xvf python-model-infer.tgz
cd -

echo ${PADDLE_ROOT}
#python3.7 -m pip install ${PADDLE_ROOT}/build/python/dist/*-cp37-cp37m-linux_x86_64.whl;
#python3.7 -m pip install nose

if [ -d "result" ];then rm -rf result
fi
mkdir result

export CUDA_VISIBLE_DEVICES=0
cases="test_resnet_fluid"

for file in ${cases}
do
    echo " "
    echo "\033[33m ====> ${file} case start \033[0m"
    python3.7 -m nose -s -v --with-xunit --xunit-file=result/${file}.xml ${file}.py
    echo "\033[33m ====> ${file} case finish \033[0m"
    echo " "
done

model_cases="test_inference_cpu \
             test_inference_gpu \
             test_inference_mkldnn \
             test_inference_trt_fp32"

export project_path
echo -e "\033[33m project_path is : ${project_path} \033[0m"
cd tests
if [ -d "result" ];then rm -rf result
fi
mkdir result
for file in ${model_cases}
do
    
    echo " "
    echo -e "\033[33m ====> ${file} case start \033[0m"
    python3.7 -m nose -s -v --with-xunit --xunit-file=result/${file}.xml ${file}.py
    echo -e "\033[33m ====> ${file} case finish \033[0m"
    echo " "
done
cd -

echo -e "\033[33m finish all cases \033[0m"
