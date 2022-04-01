#!/bin/bash
set -x
export PYTHONPATH=`pwd`:$PYTHONPATH

project_path=`pwd`
export project_path
echo -e "\033[33m project_path is : ${project_path} \033[0m"
cd ${project_path}

gpu_cases=`find ./tests -name "test*_gpu.py" | sort`
trt_fp32_cases=`find ./tests -name "test*_trt_fp32.py" | sort`
cases="${gpu_cases} ${trt_fp32_cases}"
ignore="test_bert_emb_v1_trt_fp32.py \
        test_rec_chinese_common_train_trt_fp32.py \
        test_det_mv3_east_trt_fp32.py \
        test_hub_ernie_trt_fp32.py \
        test_det_mv3_db_trt_fp32.py \
        test_v1_yolov3_r50vd_trt_fp32.py"
# disable test_det_mv3_db_trt_fp32.py since model precision not correct

# download Data
if [ -d "Data" ];then rm -rf Data
fi
# download data with numpy
wget --no-proxy -q https://sys-p0.bj.bcebos.com/inference/Data.tgz --no-check-certificate
tar -xvf Data.tgz
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


if [ -d "result" ];then rm -rf report
fi
report_path="${project_path}/report"
mkdir -p ${report_path}

for file in ${cases}
do
    test_case=`basename ${file}`
    test_case_path=`dirname ${file}`
    cd ${test_case_path}
    echo -e "\033[33m ====> ${test_case} case start \033[0m"
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        # python -m nose ${test_case} --with-allure --logdir=${report_path}
        python -m pytest ${test_case}
    fi
    echo -e "\033[33m ====> ${test_case} case finish \033[0m"
    echo " "
    cd -
done
