#!/bin/bash
export RED='\033[0;31m' # red color
export NC='\033[0m' # no color
export YELLOW='\033[33m' # yellow color

test_gpu(){
    exe_bin=$1 # ./build/clas_benchmark
    model_name=$2
    model_path=$3
    params_path=$4

    image_shape="3,224,224"
    if [ $# -ge 5 ]; then
        image_shape=$5
    fi
    printf "${YELLOW} ${model_name} input image_shape = ${image_shape} ${NC} \n";
    accuracy=1e-5;
    use_gpu=true;

    for batch_size in "1" "2" "4"
    do
        echo " "
        printf "start ${YELLOW} ${model_name}, use_gpu: ${use_gpu}, batch_size: ${batch_size}${NC}\n"
        $OUTPUT_BIN/${exe_bin} --model_name=${model_name} \
                               --model_path=${model_path} \
                               --params_path=${params_path} \
                               --image_shape=${image_shape} \
                               --baatch_size=${batch_size} \
                               --use_gpu=${use_gpu} 2>&1 | tee ${LOG_ROOT}/${model_name}_gpu_bz${batch_size}_infer.log
        printf "finish ${RED} ${model_name}, use_gpu: ${use_gpu}, batch_size: ${batch_size}${NC}\n"
        echo " "
    done                               
}


test_trt(){
    exe_bin=$1 # ./build/clas_benchmark
    model_name=$2
    model_path=$3
    params_path=$4

    image_shape="3,224,224"
    if [ $# -ge 5 ]; then
        image_shape=$5
    fi
    printf "${YELLOW} ${model_name} input image_shape = ${image_shape} ${NC} \n";
    accuracy=1e-5;
    use_gpu=true;
    use_trt=true;

    # Tesla T4 can run fp16
    if $gpu_type == "T4"; then
        declare -a trt_precisions=("fp32" "fp16")
    else
        declare -a trt_precisions=("fp32")
    fi

    for batch_size in "1" "2" "4"
    do
        for trt_precision in "${trt_precisions[@]}"  # "fp32" "fp16" "int8"
        do
            echo " "
            printf "start ${YELLOW} ${model_name}, use_trt: ${use_trt}, trt_precision: ${trt_precision}, batch_size: ${batch_size}${NC}\n"
            $OUTPUT_BIN/${exe_bin} --model_name=${model_name} \
                                --model_path=${model_path} \
                                --params_path=${params_path} \
                                --image_shape=${image_shape} \
                                --batch_size=${batch_size} \
                                --use_gpu=${use_gpu} \
                                --use_trt=${use_trt} 2>&1 | ${LOG_ROOT}/${model_name}_gpu_bz${batch_size}_infer.log
            printf "finish ${RED} ${model_name}, use_trt: ${use_trt}, trt_precision: ${trt_precision}, batch_size: ${batch_size}${NC}\n"
            echo " "
        done
    done                               
}

main(){
    printf "${YELLOW} ==== start benchmark ==== ${NC} \n"
    model_root=$1

    class_model="AlexNet \
                 DarkNet53 \
                 DenseNet121 \
                 DPN68 \
                 EfficientNetB0 \
                 GhostNet_x1_3 \
                 GoogLeNet \
                 HRNet_W18_C \
                 InceptionV4 \
                 MobileNetV1 \
                 MobileNetV2 \
                 MobileNetV3_large_x1_0 \
                 RegNetX_4GF \
                 Res2Net50_26w_4s \
                 ResNeSt50_fast_1s1x64d \
                 ResNet50 \
                 ResNet50_vd \
                 SE_ResNeXt50_vd_32x4d \
                 ShuffleNetV2 \
                 SqueezeNet1_0 \
                 VGG11 \
                 Xception41"
    
    for tests in ${class_model}
    do
        test_gpu "clas_benchmark" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/params
    
        test_trt "clas_benchmark" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/params
    done

    # ssdlite_mobilenet_v3_large
    model_case="ssdlite_mobilenet_v3_large"
    test_gpu "clas_benchmark" "${model_case}" \
             "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__model__" \
             "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__params__" \
             "3,320,320"

    test_trt "clas_benchmark" "${model_case}" \
             "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__model__" \
             "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__params__" \
             "3,320,320"
    
    # ssd_mobilenet_v1_voc
    model_case="ssd_mobilenet_v1_voc"
    test_gpu "clas_benchmark" "${model_case}" \
             "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__model__" \
             "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__params__" \
             "3,300,300"

    test_trt "clas_benchmark" "${model_case}" \
             "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__model__" \
             "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__params__" \
             "3,300,300"


    printf "${YELLOW} ==== finish benchmark ==== ${NC} \n"
}

model_root=${DATA_ROOT}/PaddleClas/infer_static
if [ $# -ge 1 ]; then
    model_root=$1
fi

main ${model_root}
