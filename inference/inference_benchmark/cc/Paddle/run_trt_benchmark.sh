#!/bin/bash
export RED='\033[0;31m' # red color
export NC='\033[0m' # no color
export YELLOW='\033[33m' # yellow color

predict_trt(){
    # predict trt
    exe_bin=$1 # ./build/clas_benchmark
    model_name=$2
    model_path=$3
    params_path=$4

    image_shape="3,224,224"
    if [ $# -ge 5 ]; then
        image_shape=$5
    fi
    printf "${RED} ${model_name} input image_shape = ${image_shape} ${NC} \n";

    trt_min_subgraph_size="3"
    if [ $# -ge 6 ]; then
        trt_min_subgraph_size=$6
    fi

    model_type="static"
    log_root="./logs/${model_name}_trt"

    gpu_type=`nvidia-smi -q | grep "Product Name" | head -n 1 | awk '{print $NF}'`

    # Tesla T4 can run fp16
    if $gpu_type == "T4"; then
        declare -a trt_precisions=("fp32" "fp16")
    else
        declare -a trt_precisions=("fp32")
    fi
    use_trt=True

    for batch_size in "1" "2" "4"
    do
        for trt_precision in "${trt_precisions[@]}"  # "fp32" "fp16" "int8"
        do
            echo " "
            printf "start ${RED} ${model_name}, use_gpu: True, \
    use_trt: ${use_trt}, trt_precision: ${trt_precision}, batch_size: ${batch_size}${NC}\n"
            log_path="${log_root}/use_trt_${use_trt}/trt_precision_${trt_precision}/bz_${batch_size}"
            mkdir -p ${log_path}

            ./build/${exe_bin} --model_name=${model_name} \
                                --model_path=${model_path} \
                                --params_path=${params_path} \
                                --use_gpu=True \
                                --image_shape=${image_shape} \
                                --repeats=1000 \
                                --batch_size=${batch_size} \
                                --use_trt=${use_trt} \
                                --trt_min_subgraph_size=${trt_min_subgraph_size} \
                                --trt_precision=${trt_precision} 2>&1 | tee ${log_path}/${model_name}_infer.log
            printf "finish ${RED} ${model_name}, use_gpu: True, \
    use_trt: ${use_trt}, trt_precision: ${trt_precision}, batch_size: ${batch_size}${NC} \n"
            echo " "
        done
    done
}


main(){
    printf "${YELLOW} ==== start benchmark ==== ${NC} \n"
    model_root=$1

    # mobilenet_v2-paddle
    models="mobilenet_v2-paddle"
    predict_trt clas_benchmark ${models} ${model_root}/${models}/model ${model_root}/${models}/params

    # resnet101-paddle
    models="resnet101-paddle"
    predict_trt clas_benchmark ${models} ${model_root}/${models} ""

    # mobilenet_ssd-paddle
    models="mobilenet_ssd-paddle"
    predict_trt clas_benchmark ${models} ${model_root}/${models}/model "${model_root}/${models}/params" "3,300,300"

    # faster_rcnn_r50_1x-paddle, trt_min_subgraph_size=40
    models="faster_rcnn_r50_1x-paddle"
    predict_trt rcnn_benchmark ${models} ${model_root}/${models}/__model__ \
                               "${model_root}/${models}/__params__" \
                               "3,640,640" "40"

    # deeplabv3p_xception_769_fp32-paddle
    models="deeplabv3p_xception_769_fp32-paddle"
    predict_trt clas_benchmark ${models} ${model_root}/${models}/model "${model_root}/${models}/params" "3,769,769"

    # unet-paddle
    models="unet-paddle"
    predict_trt clas_benchmark ${models} ${model_root}/${models}/model \
                                         ${model_root}/${models}/params \
                                         "3,512,512"

    # bert_emb_v1-paddle
    models="bert_emb_v1-paddle"
    predict_trt bert_benchmark ${models} ${model_root}/${models} ""

    printf "${YELLOW} ==== finish benchmark ==== ${NC} \n"
}

model_root=${PWD}/Data/Paddle
if [ $# -ge 1 ]; then
    model_root=$1
fi

main ${model_root}
