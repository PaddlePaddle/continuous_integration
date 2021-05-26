#!/bin/bash
export RED='\033[0;31m' # red color
export NC='\033[0m' # no color
export YELLOW='\033[33m' # yellow color

function test_cpu(){
    exe_bin=$1 # ./build/clas_benchmark
    model_name=$2
    model_path=$3
    params_path=$4
    declare -a name_batch_size=$5[@]
    cpu_batch_size=(${!name_batch_size})
    image_shape="3,224,224"
    if [ $# -ge 6 ]; then
        image_shape=$6
    fi
    printf "${YELLOW} ${model_name} input image_shape = ${image_shape} ${NC} \n";
    use_gpu=false;

    for batch_size in ${cpu_batch_size[@]}
    do
        echo " "
        printf "start ${YELLOW} ${model_name}, use_gpu: ${use_gpu}, batch_size: ${batch_size}${NC}\n"

        log_file="${LOG_ROOT}/${model_name}_cpu_bz${batch_size}_infer.log"
        $OUTPUT_BIN/${exe_bin} --model_name=${model_name} \
            --model_path=${model_path} \
            --params_path=${params_path} \
            --image_shape=${image_shape} \
            --batch_size=${batch_size} \
            --model_type=${MODEL_TYPE} \
            --repeats=500 \
            --use_gpu=${use_gpu} >> ${log_file} 2>&1 | python3.7 ${CASE_ROOT}/py_mem.py "$OUTPUT_BIN/${exe_bin}" >> ${log_file} 2>&1
        printf "finish ${RED} ${model_name}, use_gpu: ${use_gpu}, batch_size: ${batch_size}${NC}\n"
        echo " "
    done                               
}


function test_mkldnn(){
    exe_bin=$1 # ./build/clas_benchmark
    model_name=$2
    model_path=$3
    params_path=$4
    declare -a name_batch_size=$5[@]
    cpu_batch_size=(${!name_batch_size})
    declare -a name_threads=$6[@]
    cpu_num_threads=(${!name_threads})
    image_shape="3,224,224"
    if [ $# -ge 7 ]; then
        image_shape=$7
    fi
    use_interpolate_mkldnn_pass=false;
    if [ $# -ge 8 ]; then
        use_interpolate_mkldnn_pass=$8
    fi

    printf "${YELLOW} ${model_name} input image_shape = ${image_shape} ${NC} \n";
    use_gpu=false;
    use_mkldnn=true;

    for batch_size in ${cpu_batch_size[@]}
    do
        for cpu_math_library_num_threads in ${cpu_num_threads[@]}
        do
            echo " "
            printf "start ${YELLOW} ${model_name}, use_mkldnn: ${use_mkldnn}, cpu_math_library_num_threads: ${cpu_math_library_num_threads}, batch_size: ${batch_size}${NC}\n"

            log_file="${LOG_ROOT}/${model_name}_mkldnn_${cpu_math_library_num_threads}_bz${batch_size}_infer.log"
            $OUTPUT_BIN/${exe_bin} --model_name=${model_name} \
                --model_path=${model_path} \
                --params_path=${params_path} \
                --image_shape=${image_shape} \
                --batch_size=${batch_size} \
                --use_gpu=${use_gpu} \
                --repeats=500 \
                --model_type=${MODEL_TYPE} \
                --use_interpolate_mkldnn_pass=${use_interpolate_mkldnn_pass} \
                --cpu_math_library_num_threads=${cpu_math_library_num_threads} \
                --use_mkldnn_=${use_mkldnn} >> ${log_file} 2>&1 | python3.7 ${CASE_ROOT}/py_mem.py "$OUTPUT_BIN/${exe_bin}" >> ${log_file} 2>&1

            printf "finish ${RED} ${model_name}, use_mkldnn: ${use_mkldnn}, cpu_math_library_num_threads: ${cpu_math_library_num_threads}, batch_size: ${batch_size}${NC}\n"
            echo " "
        done
    done                               
}

function run_clas_mkl_func(){
    model_root=${DATA_ROOT}/PaddleClas/infer_static
    if [ $# -ge 1 ]; then
        model_root=$1
    fi
    printf "${YELLOW} ==== start benchmark ==== ${NC} \n"
    declare name_batch_size=$2[@]
    cpu_batch_size=(${!name_batch_size})
    declare name_threads=$3[@]
    cpu_num_threads=(${!name_threads})
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
        test_cpu "clas_benchmark" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/params cpu_batch_size \
        
        test_mkldnn "clas_benchmark" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/params cpu_batch_size cpu_num_threads
    done

    if [ "${MODEL_TYPE}" == "static" ]; then
        # ssdlite_mobilenet_v3_large
        model_case="ssdlite_mobilenet_v3_large"
        test_cpu "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__model__" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__params__" \
                cpu_batch_size "3,320,320"

        test_mkldnn "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__model__" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__params__" \
                cpu_batch_size cpu_num_threads "3,320,320"
        
        # ssd_mobilenet_v1_voc
        model_case="ssd_mobilenet_v1_voc"
        test_cpu "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__model__" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__params__" \
                cpu_batch_size "3,300,300"

        test_mkldnn "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__model__" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__params__" \
                cpu_batch_size cpu_num_threads "3,300,300"
        
        seg_model="deeplabv3p \
                fastscnn \
                hrnet \
                icnet \
                pspnet \
                unet"

        for tests in ${seg_model}
        do
            test_cpu "clas_benchmark" "${tests}" \
                    ${DATA_ROOT}/PaddleSeg/infer_static/${tests}/__model__ \
                    ${DATA_ROOT}/PaddleSeg/infer_static/${tests}/__params__ \
                    cpu_batch_size "3,512,512"
        
            test_mkldnn "clas_benchmark" "${tests}" \
                    ${DATA_ROOT}/PaddleSeg/infer_static/${tests}/__model__ \
                    ${DATA_ROOT}/PaddleSeg/infer_static/${tests}/__params__ \
                    cpu_batch_size cpu_num_threads "3,512,512" "true"
        done

        # ch_ppocr_mobile_v1.1_cls_infer
        model_case="ch_ppocr_mobile_v1.1_cls_infer"
        test_cpu "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/model" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/params" \
                cpu_batch_size "3,48,192"

        test_mkldnn "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/model" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/params" \
                cpu_batch_size cpu_num_threads "3,48,192"
        
        # ch_ppocr_mobile_v1.1_det_infer
        model_case="ch_ppocr_mobile_v1.1_det_infer"
        test_cpu "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/model" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/params" \
                cpu_batch_size "3,640,640"

        test_mkldnn "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/model" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/params" \
                cpu_batch_size cpu_num_threads "3,640,640" "true"
        
        # ch_ppocr_mobile_v1.1_rec_infer
        model_case="ch_ppocr_mobile_v1.1_rec_infer"
        test_cpu "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/model" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/params" \
                cpu_batch_size "3,32,320"

        test_mkldnn "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/model" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/params" \
                cpu_batch_size cpu_num_threads "3,32,320" "10"
    fi

    printf "${YELLOW} ==== finish benchmark ==== ${NC} \n"
}
