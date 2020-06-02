#!/bin/bash

project_path=$(cd "$(dirname "$0")";pwd)

if [ -d "Data" ];then rm -rf Data
fi
mkdir -p ./Data
cd ./Data
wget https://sys-p0.bj.bcebos.com/inference/python-infer.tgz --no-check-certificate
tar -xvf python-infer.tgz
wget https://sys-p0.bj.bcebos.com/inference/python-model-infer.tgz --no-check-certificate
tar -xvf python-model-infer.tgz
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

class_models="MobileNetV1_pretrained \
              ResNet50_pretrained \
              SE_ResNeXt50_32x4d_pretrained \
              Xception_41_pretrained"

detect_models="blazeface_nas_128 \
               faster_rcnn_r50_1x \
               mask_rcnn_r50_1x \
               yolov3_darknet"

seg_models="deeplabv3_mobilenetv2"

cd tests
for model in ${class_models}
do
    echo "====> ${model} case start"
    model_path=${project_path}/Data/python-model-infer/classification
    python test_inference_cpu.py --model_path=${model_path}/${model}/model \
                                       --data_path=${model_path}/${model}/data/data.json
    echo "====> ${model} case finish"
done

for model in ${detect_models}
do
    echo "====> ${model} case start"
    model_path=${project_path}/Data/python-model-infer/Detection
    python test_inference_cpu.py --model_path=${model_path}/${model}/model \
                                       --data_path=${model_path}/${model}/data/data.json
    echo "====> ${model} case finish"
done

for model in ${seg_models}
do
    echo "====> ${model} case start"
    model_path=${project_path}/Data/python-model-infer/segmentation
    python test_inference_cpu.py --model_path=${model_path}/${model}/model \
                                       --data_path=${model_path}/${model}/data/data.json
    echo "====> ${model} case finish"
done
cd -
