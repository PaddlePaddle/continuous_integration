#!/bin/bash

if [ -d "Data" ];then rm -rf Data
fi
mkdir -p ./Data
cd ./Data
wget https://sys-p0.bj.bcebos.com/inference/python-infer.tgz
tar -xvf python-infer.tgz
cd -

echo ${PADDLE_ROOT}
python -m pip install ${PADDLE_ROOT}/build/python/dist/*-cp37-cp37m-linux_x86_64.whl;
python -m pip install nose

if [ -d "result" ];then rm -rf result
fi
mkdir result

export CUDA_VISIBLE_DEVICES=0
cases="test_resnet_fluid.py"

for file in ${cases}
do
    echo "====> ${file} case start"
    python -m nose -s -v --with-xunit --xunit-file=result/${file}_single.xml ${file}
    echo "====> ${file} case finish"
done
