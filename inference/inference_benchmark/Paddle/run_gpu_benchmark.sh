#!/bin/bash
export RED='\033[0;31m' # red color
export NC='\033[0m' # no color
export YELLOW='\033[33m' # yellow color

predict_gpu(){
    # predict gpu
    exe_bin=$1 # ./build/clas_benchmark
    model_name=$2
    model_path=$3
    params_path=$4

    image_shape="3,224,224"
    if [ $# -ge 5 ]; then
        image_shape=$5
    fi
    printf "${RED} ${model_name} input image_shape = ${image_shape} ${NC} \n";

    model_type="static"
    log_root="./logs/${model_name}_gpu"

    for batch_size in "1" "2" "4"
    do
        echo " "
        printf "start ${RED} ${model_name}, use_gpu: True, batch_size: ${batch_size}${NC}\n"
        log_path="${log_root}/bz_${batch_size}"
        mkdir -p ${log_path}

        ./build/${exe_bin} --model_name=${model_name} \
                            --model_path=${model_path} \
                            --params_path=${params_path} \
                            --image_shape=${image_shape} \
                            --batch_size=${batch_size} \
                            --use_gpu=true \
                            --repeats=1000 2>&1 | tee ${log_path}/${model_name}_infer.log
        printf "finish ${RED} ${model_name}, use_gpu: True, batch_size: ${batch_size}${NC}\n"
        echo " "
    done
}


main(){
    printf "${YELLOW} ==== start benchmark ==== ${NC} \n"
    model_root=$1

    # mobilenet_v2-paddle
    models="mobilenet_v2-paddle"
    predict_gpu clas_benchmark ${models} ${model_root}/${models}/model ${model_root}/${models}/params

    # resnet101-paddle
    models="resnet101-paddle"
    predict_gpu clas_benchmark ${models} ${model_root}/${models} ""

    # mobilenet_ssd-paddle
    models="mobilenet_ssd-paddle"
    predict_gpu clas_benchmark ${models} ${model_root}/${models}/model "${model_root}/${models}/params" "3,300,300"

    # faser_rcnn_r50_1x-paddle
    models="faser_rcnn_r50_1x-paddle"
    predict_gpu rcnn_benchmark ${models} ${model_root}/${models}/__model__ "${model_root}/${models}/__params__" "3,640,640"

    # deeplabv3p_xception_769_fp32-paddle
    models="deeplabv3p_xception_769_fp32-paddle"
    predict_gpu clas_benchmark ${models} ${model_root}/${models}/model "${model_root}/${models}/params" "3,769,769"

    # unet-paddle
    models="unet-paddle"
    predict_gpu clas_benchmark ${models} ${model_root}/${models} "" "1,512,512"

    # bert_emb_v1-paddle
    models="bert_emb_v1-paddle"
    predict_gpu bert_benchmark ${models} ${model_root}/${models} ""

    printf "${YELLOW} ==== finish benchmark ==== ${NC} \n"
}

model_root=${PWD}/Data/Paddle
if [ $# -ge 1 ]; then
    model_root=$1
fi

main ${model_root}
