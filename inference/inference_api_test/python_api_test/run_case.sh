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
# download ocr models
wget -q https://sys-p0.bj.bcebos.com/inference/python-ocr-infer.tgz --no-check-certificate
tar -xvf python-ocr-infer.tgz
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

declare -A ModelCase
ModelCase["cpu"]"test_blazeface_cpu \
                 test_cpu_helper \
                 test_deeplabv3_cpu \
                 test_faster_rcnn_cpu \
                 test_mask_rcnn_cpu \
                 test_mobilenetv1_cpu \
                 test_resnet50_cpu \
                 test_seresnext50_cpu \
                 test_xception41_cpu \
                 test_yolov3_cpu"

ModelCase["gpu"]="test_blazeface_gpu \
                  test_gpu_helper \
                  test_deeplabv3_gpu \
                  test_faster_rcnn_gpu \
                  test_mask_rcnn_gpu \
                  test_mobilenetv1_gpu \
                  test_resnet50_gpu \
                  test_seresnext50_gpu \
                  test_xception41_gpu \
                  test_yolov3_gpu"

ModelCase["mkldnn"]="test_blazeface_mkldnn \
                  test_mkldnn_helper \
                  test_deeplabv3_mkldnn \
                  test_faster_rcnn_mkldnn \
                  test_mask_rcnn_mkldnn \
                  test_mobilenetv1_mkldnn \
                  test_resnet50_mkldnn \
                  test_seresnext50_mkldnn \
                  test_xception41_mkldnn \
                  test_yolov3_mkldnn \
                  test_det_mv3_db_mkldnn"

ModelCase["trt_fp32"]="test_blazeface_trt_fp32 \
                  test_trt_fp32_helper \
                  test_deeplabv3_trt_fp32 \
                  test_faster_rcnn_trt_fp32 \
                  test_mask_rcnn_trt_fp32 \
                  test_mobilenetv1_trt_fp32 \
                  test_resnet50_trt_fp32 \
                  test_seresnext50_trt_fp32 \
                  test_xception41_trt_fp32 \
                  test_yolov3_trt_fp32"

export project_path
echo -e "\033[33m project_path is : ${project_path} \033[0m"
cd tests
if [ -d "result" ];then rm -rf result
fi
result_path="${project_path}/tests/result"
mkdir result

for config in "cpu" "gpu" "mkldnn" "trt_fp32"
do
    cd ${config}
    for file in ${ModelCase[${config}]}
    do
        echo " "
        echo -e "\033[33m ====> ${file} case start \033[0m"
        python3.7 -m nose -s -v --with-xunit --xunit-file=${result_path}/${file}.xml ${file}.py
        echo -e "\033[33m ====> ${file} case finish \033[0m"
        echo " "
    done
    cd - # back tests
done

cd .. # back ../tests

echo -e "\033[33m finish all cases \033[0m"
