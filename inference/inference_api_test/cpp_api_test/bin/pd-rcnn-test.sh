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

    image_shape="3,640,640"
    if [ $# -ge 6 ]; then
        image_shape=$6
    fi
    printf "${YELLOW} ${model_name} input image_shape = ${image_shape} ${NC} \n";

    use_gpu=true;

    for batch_size in "1" "2" "4"
    do
        echo " "
        printf "start ${YELLOW} ${model_name}, use_gpu: ${use_gpu}, batch_size: ${batch_size}${NC}\n"
        $OUTPUT_BIN/${exe_bin} --model_name=${model_name} \
                               --model_path=${model_path} \
                               --params_path=${params_path} \
                               --batch_size=${batch_size} \
                               --use_gpu=${use_gpu} \
                               --accuracy=${accuracy} \
                               --image_shape=${image_shape} \
                               --gtest_output=xml:test_${model_name}_gpu_${accuracy}_bz${batch_size}.xml
        python3.7 ${CASE_ROOT}/py_sed.py --input_file=test_${model_name}_gpu_${accuracy}_bz${batch_size}.xml \
                                      --testsuite_old_name="test_rcnn_model"
        printf "finish ${RED} ${model_name}, use_gpu: ${use_gpu}, batch_size: ${batch_size}${NC}\n"
        echo " "
    done                               
}


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

    for batch_size in "1" "2" "4"
    do
        echo " "
        printf "start ${YELLOW} ${model_name}, use_trt: ${use_trt}, batch_size: ${batch_size}${NC}\n"
        $OUTPUT_BIN/${exe_bin} --model_name=${model_name} \
                               --model_path=${model_path} \
                               --params_path=${params_path} \
                               --batch_size=${batch_size} \
                               --use_gpu=${use_gpu} \
                               --use_trt=${use_trt} \
                               --accuracy=${accuracy} \
                               --image_shape=${image_shape} \
                               --gtest_output=xml:test_${model_name}_trt_${accuracy}_bz${batch_size}.xml
        python3.7 ${CASE_ROOT}/py_sed.py --input_file=test_${model_name}_trt_${accuracy}_bz${batch_size}.xml \
                                      --testsuite_old_name="test_rcnn_model"
        printf "finish ${RED} ${model_name}, use_trt: ${use_trt}, batch_size: ${batch_size}${NC}\n"
        echo " "
    done                               
}

main(){
    printf "${YELLOW} ==== start benchmark ==== ${NC} \n"
    model_root=$1

    rcnn_model="mask_rcnn_r50_1x"
                
    for tests in ${rcnn_model}
    do
        test_gpu "test_rcnn_model" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/__params__
    
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
