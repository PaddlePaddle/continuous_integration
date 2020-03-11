#!/bin/bash

mkdir -p ./Data
cd ./Data
wget https://sys-p0.bj.bcebos.com/inference/python-infer.tgz
tar -xvf python-infer.tgz
cd -

rm -rf result
mkdir result
export CUDA_VISIBLE_DEVICES=0

cases="test_resnet_fluid.py"

for file in ${cases}
do
    echo "====> ${file} case start"
    python -m nose -s -v --with-xunit --xunit-file=result/${file}_single.xml ${file}
    echo "====> ${file} case finish"
done
