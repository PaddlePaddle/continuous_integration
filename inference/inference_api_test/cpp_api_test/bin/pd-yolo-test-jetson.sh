#!/bin/bash
export RED='\033[0;31m' # red color
export NC='\033[0m' # no color
export YELLOW='\033[33m' # yellow color

test_gpu(){
    exe_bin=$1 # ./build/clas_benchmark
    model_name=$2
    model_path=$3
    params_path=$4

    accuracy="1e-5"
    if [ $# -ge 5 ]; then
        accuracy=$5
    fi

    printf "${YELLOW} ${model_name} ${NC} \n";
    use_gpu=true;

    for batch_size in "1" "2"
    do
        echo " "
        printf "start ${YELLOW} ${model_name}, use_gpu: ${use_gpu}, batch_size: ${batch_size}${NC}\n"
        $OUTPUT_BIN/${exe_bin} --model_name=${model_name} \
                               --model_path=${model_path} \
                               --params_path=${params_path} \
                               --batch_size=${batch_size} \
                               --use_gpu=${use_gpu} \
                               --accuracy=${accuracy} \
                               --gtest_output=xml:test_${model_name}_gpu_${accuracy}_bz${batch_size}.xml
        python3.6 ${CASE_ROOT}/py_sed.py --input_file=test_${model_name}_gpu_${accuracy}_bz${batch_size}.xml \
                                      --testsuite_old_name="test_yolo_model"
        printf "finish ${RED} ${model_name}, use_gpu: ${use_gpu}, batch_size: ${batch_size}${NC}\n"
        echo " "
    done                               
}


test_trt(){
    exe_bin=$1 # ./build/clas_benchmark
    model_name=$2
    model_path=$3
    params_path=$4

    printf "${YELLOW} ${model_name} ${NC} \n";
    accuracy="1e-5"
    if [ $# -ge 5 ]; then
        accuracy=$5
    fi

    use_gpu=true;
    use_trt=true;

    # Tesla T4 can run fp16
    if [ "$gpu_type" == "xavier" ]; then
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
                accuracy=9e-1
            else
                accuracy=3e-4
            fi
            printf "${YELLOW} ${trt_precision} accuracy set to ${accuracy} ${NC}\n"
            $OUTPUT_BIN/${exe_bin} --model_name=${model_name} \
                                --model_path=${model_path} \
                                --params_path=${params_path} \
                                --batch_size=${batch_size} \
                                --use_gpu=${use_gpu} \
                                --use_trt=${use_trt} \
                                --accuracy=${accuracy} \
                                --trt_precision=${trt_precision} \
                                --gtest_output=xml:test_${model_name}_trt_${trt_precision}_${accuracy}_bz${batch_size}.xml
            python3.6 ${CASE_ROOT}/py_sed.py --input_file=test_${model_name}_trt_${trt_precision}_${accuracy}_bz${batch_size}.xml \
                                        --testsuite_old_name="test_yolo_model"
            printf "finish ${RED} ${model_name}, use_trt: ${use_trt}, trt_precision: ${trt_precision}, batch_size: ${batch_size}${NC}\n"
            echo " "
        done
    done                               
}

main(){
    printf "${YELLOW} ==== start benchmark ==== ${NC} \n"
    model_root=$1

    yolo_model="yolov3_r50vd_dcn"
                
    for tests in ${yolo_model}
    do
        test_gpu "test_yolo_model" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/__params__ 2e-4
    
        test_trt "test_yolo_model" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/__params__ 2e-5
    done

    printf "${YELLOW} ==== finish benchmark ==== ${NC} \n"
}

model_root=${DATA_ROOT}/PaddleDetection/infer_static
if [ $# -ge 1 ]; then
    model_root=$1
fi

main ${model_root}
