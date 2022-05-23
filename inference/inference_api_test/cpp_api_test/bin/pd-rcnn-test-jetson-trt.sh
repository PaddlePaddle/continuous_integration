#!/bin/bash
export RED='\033[0;31m' # red color
export NC='\033[0m' # no color
export YELLOW='\033[33m' # yellow color

test_trt(){
    exe_bin=$1 # ./build/clas_benchmark
    model_name=$2
    model_path=$3
    params_path=$4

    accuracy="1e-5"
    if [ $# -ge 5 ]; then
        accuracy=$5
    fi

    image_shape="3,640,640"
    if [ $# -ge 6 ]; then
        image_shape=$6
    fi
    printf "${YELLOW} ${model_name} input image_shape = ${image_shape} ${NC} \n";

    use_gpu=true;
    use_trt=true;
    trt_min_subgraph_size=40;  # rcnn model need set to 40

    # Tesla T4 can run fp16
    if [ "$gpu_type" == "T4" ]; then
        declare -a trt_precisions=("fp32" "fp16")
    else
        declare -a trt_precisions=("fp32")
    fi

    for batch_size in "1" "2"
    do  
        for trt_precision in "${trt_precisions[@]}"  # "fp32" "fp16" "int8"
        do
            echo " "
            printf "start ${YELLOW} ${model_name}, use_trt: ${use_trt}, trt_precision: ${trt_precision}, batch_size: ${batch_size}${NC}\n"
            if [ $trt_precision == "fp16" ]; then
                accuracy=1e-3
            else
                accuracy=1e-5
            fi
            printf "${YELLOW} ${trt_precision} accuracy set to ${accuracy} ${NC}\n"
            $OUTPUT_BIN/${exe_bin} --model_name=${model_name} \
                                    --model_path=${model_path} \
                                    --params_path=${params_path} \
                                    --batch_size=${batch_size} \
                                    --use_gpu=${use_gpu} \
                                    --use_trt=${use_trt} \
                                    --accuracy=${accuracy} \
                                    --trt_min_subgraph_size=${trt_min_subgraph_size} \
                                    --trt_precision=${trt_precision} \
                                    --image_shape=${image_shape} \
                                    --gtest_output=xml:test_${model_name}_trt_${trt_precision}_${accuracy}_bz${batch_size}.xml
            python3.6 ${CASE_ROOT}/py_sed.py --input_file=test_${model_name}_trt_${trt_precision}_${accuracy}_bz${batch_size}.xml \
                                            --testsuite_old_name="test_rcnn_model"
            printf "finish ${RED} ${model_name}, use_trt: ${use_trt}, trt_precision: ${trt_precision}, batch_size: ${batch_size}${NC}\n"
            echo " "
        done
    done                               
}

main(){
    printf "${YELLOW} ==== start benchmark ==== ${NC} \n"
    model_root=$1

    rcnn_model="mask_rcnn_r50_1x \
                faster_rcnn_r50_1x \
                faster_rcnn_dcn_r50_vd_fpn_3x_server_side"
                
    for tests in ${rcnn_model}
    do
        test_trt "test_rcnn_model" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/__params__
    done

    printf "${YELLOW} ==== finish benchmark ==== ${NC} \n"
}

model_root=${DATA_ROOT}/PaddleDetection/infer_static
if [ $# -ge 1 ]; then
    model_root=$1
fi

main ${model_root}

