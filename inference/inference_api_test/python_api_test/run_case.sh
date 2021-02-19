#!/bin/bash

project_path=$(cd "$(dirname "$0")";pwd)
echo -e "\033[33m project_path is : ${project_path} \033[0m"
cd ${project_path}

if [ -d "Data" ];then rm -rf Data
fi
mkdir -p ./Data
cd ./Data
# download models
wget --no-proxy -q https://sys-p0.bj.bcebos.com/inference/python-infer.tgz --no-check-certificate
tar -xvf python-infer.tgz
wget --no-proxy -q https://sys-p0.bj.bcebos.com/inference/python-model-infer.tgz --no-check-certificate
tar -xvf python-model-infer.tgz

# download ocr models
wget --no-proxy -q https://sys-p0.bj.bcebos.com/inference/python-ocr-infer.tgz --no-check-certificate
tar -xvf python-ocr-infer.tgz

# download hub-ernie models
wget --no-proxy -q https://sys-p0.bj.bcebos.com/inference/python-hub-ernie.tgz --no-check-certificate
tar -xvf python-hub-ernie.tgz

# download paddle-slim models
wget --no-proxy -q https://sys-p0.bj.bcebos.com/inference/python-slim-infer.tgz --no-check-certificate
tar -xvf python-slim-infer.tgz
cd -

echo ${PADDLE_ROOT}
#python3.7 -m pip install ${PADDLE_ROOT}/build/python/dist/*-cp37-cp37m-linux_x86_64.whl;
#python3.7 -m pip install nose

export CUDA_VISIBLE_DEVICES=0

declare -A ModelCase
ModelCase["cpu"]="test_blazeface_cpu \
                  test_deeplabv3_cpu \
                  test_faster_rcnn_cpu \
                  test_mask_rcnn_cpu \
                  test_mobilenetv1_cpu \
                  test_resnet50_cpu \
                  test_seresnext50_cpu \
                  test_xception41_cpu \
                  test_yolov3_cpu \
                  test_det_mv3_db_cpu \
                  test_det_mv3_east_cpu \
                  test_rec_chinese_common_train_cpu \
                  test_rec_r34_vd_tps_bilstm_attn_cpu"

ModelCase["gpu"]="test_blazeface_gpu \
                  test_deeplabv3_gpu \
                  test_faster_rcnn_gpu \
                  test_mask_rcnn_gpu \
                  test_mobilenetv1_gpu \
                  test_resnet50_gpu \
                  test_seresnext50_gpu \
                  test_xception41_gpu \
                  test_yolov3_gpu \
                  test_det_mv3_db_gpu \
                  test_det_mv3_east_gpu \
                  test_rec_chinese_common_train_gpu \
                  test_hub_ernie_gpu \
                  test_rec_r34_vd_tps_bilstm_attn_gpu \
                  test_dete_dist_yolov3_v1_gpu \
                  test_dete_prune_yolov3_darknet_voc_gpu \
                  test_dete_prune_faster_rcnn_r50_1x_gpu \
                  test_dete_prune_mask_rcnn_r50_1x_gpu \
                  test_bert_emb_v1_gpu"

ModelCase["mkldnn"]="test_blazeface_mkldnn \
                     test_deeplabv3_mkldnn \
                     test_faster_rcnn_mkldnn \
                     test_mask_rcnn_mkldnn \
                     test_mobilenetv1_mkldnn \
                     test_resnet50_mkldnn \
                     test_seresnext50_mkldnn \
                     test_xception41_mkldnn \
                     test_yolov3_mkldnn \
                     test_det_mv3_db_mkldnn \
                     test_det_mv3_east_mkldnn \
                     test_rec_chinese_common_train_mkldnn \
                     test_rec_r34_vd_tps_bilstm_attn_mkldnn \
                     test_hub_ernie_mkldnn"

ModelCase["trt_fp32"]="test_blazeface_trt_fp32 \
                       test_deeplabv3_trt_fp32 \
                       test_faster_rcnn_trt_fp32 \
                       test_mask_rcnn_trt_fp32 \
                       test_mobilenetv1_trt_fp32 \
                       test_resnet50_trt_fp32 \
                       test_seresnext50_trt_fp32 \
                       test_xception41_trt_fp32 \
                       test_yolov3_trt_fp32 \
                       test_det_mv3_db_trt_fp32 \
                       "

# disabled model
"
test_det_mv3_east_trt_fp32 \
test_bert_emb_v1_trt_fp32 \
test_rec_chinese_common_train_trt_fp32
"

export project_path
echo -e "\033[33m project_path is : ${project_path} \033[0m"
cd tests
if [ -d "result" ];then rm -rf result
fi
result_path="${project_path}/tests/result"
mkdir result

for config in "cpu" "gpu" "mkldnn" "trt_fp32"
do
    cd ${project_path}/tests/${config}
    echo -e "\033[33m start ${config} tests, cd test_path(${project_path}/tests/${config}) \033[0m"
    for file in ${ModelCase[${config}]}
    do
        echo " "
        echo -e "\033[33m ====> ${file} case start \033[0m"
        python3.7 -m nose -s -v --with-xunit --xunit-file=${result_path}/${file}.xml ${file}.py
        echo -e "\033[33m ====> ${file} case finish \033[0m"
        echo " "
    done
    echo -e "\033[33m finish ${config} tests, return parent dirs \033[0m"
    cd - # back tests
done

cd .. # back ../tests

echo -e "\033[33m finish all cases \033[0m"
