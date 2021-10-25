#!/bin/bash
export RED='\033[0;31m' # red color
export NC='\033[0m' # no color
export YELLOW='\033[33m' # yellow color

gen_int8_calib(){
    exe_bin=$1 # ./build/clas_benchmark
    model_name=$2
    model_path=$3
    params_path=$4

    image_shape="3,224,224"
    if [ $# -ge 5 ]; then
        image_shape=$5
    fi
    printf "${YELLOW} ${model_name} input image_shape = ${image_shape} ${NC} \n";
    use_gpu=true;
    use_trt=true;

    trt_min_subgraph_size=10;  # ch_ppocr_mobile_v1.1_rec_infer model need set to 10
    if [ $# -ge 6 ]; then
        trt_min_subgraph_size=$6
    fi

    batch_size="1"
    trt_precision="int8"
    echo " "
    printf "start ${YELLOW} ${model_name}, use_trt: ${use_trt}, trt_precision: ${trt_precision}, batch_size: ${batch_size}${NC}\n"

    # log_file="${LOG_ROOT}/Gen_calib_${model_name}_trt_${trt_precision}_bz${batch_size}_infer.log"
    if [ "${MODEL_TYPE}" == "static_prune_op" ]; then
        echo "========================== start prune model op attribute +++++++++++++++++++++++++++"
        python3.8 ${UTILS_ROOT}/model_clip.py --model_file="${model_path}" \
                                              --params_file="${params_path}" \
                                              --output_model_path="${DATA_ROOT}/prune_model/${model_name}/inference"
        model_path="${DATA_ROOT}/prune_model/${model_name}/inference.pdmodel"
        params_path="${DATA_ROOT}/prune_model/${model_name}/inference.pdiparams"
    fi;
    $OUTPUT_BIN/${exe_bin} --model_name=${model_name} \
        --model_path=${model_path} \
        --params_path=${params_path} \
        --image_shape=${image_shape} \
        --batch_size=${batch_size} \
        --use_gpu=${use_gpu} \
        --model_type=${MODEL_TYPE} \
        --trt_precision=${trt_precision} \
        --trt_min_subgraph_size=${trt_min_subgraph_size} \
        --repeats=1 \
        --warmup_times=1 \
        --use_trt=${use_trt}

    printf "finish ${RED} ${model_name}, use_trt: ${use_trt}, trt_precision: ${trt_precision}, batch_size: ${batch_size}${NC}\n"
    echo " "                          
}

test_int8(){
    exe_bin=$1 # ./build/clas_benchmark
    model_name=$2
    model_path=$3
    params_path=$4

    image_shape="3,224,224"
    if [ $# -ge 5 ]; then
        image_shape=$5
    fi
    printf "${YELLOW} ${model_name} input image_shape = ${image_shape} ${NC} \n";
    use_gpu=true;
    use_trt=true;

    trt_min_subgraph_size=10;  # ch_ppocr_mobile_v1.1_rec_infer model need set to 10
    if [ $# -ge 6 ]; then
        trt_min_subgraph_size=$6
    fi

    for batch_size in "1" "2" "4"
    do
        trt_precision="int8"
        echo " "
        printf "start ${YELLOW} ${model_name} generate calib, use_trt: ${use_trt}, trt_precision: ${trt_precision}, batch_size: ${batch_size}${NC}\n"

        log_file="${LOG_ROOT}/${model_name}_trt_${trt_precision}_bz${batch_size}_infer.log"
        if [ "${MODEL_TYPE}" == "static_prune_op" ]; then
            echo "========================== start prune model op attribute +++++++++++++++++++++++++++"
            python3.8 ${UTILS_ROOT}/model_clip.py --model_file="${model_path}" \
                                                  --params_file="${params_path}" \
                                                  --output_model_path="${DATA_ROOT}/prune_model/${model_name}/inference"
            model_path="${DATA_ROOT}/prune_model/${model_name}/inference.pdmodel"
            params_path="${DATA_ROOT}/prune_model/${model_name}/inference.pdiparams"
        fi;
        $OUTPUT_BIN/${exe_bin} --model_name=${model_name} \
            --model_path=${model_path} \
            --params_path=${params_path} \
            --image_shape=${image_shape} \
            --batch_size=${batch_size} \
            --use_gpu=${use_gpu} \
            --model_type=${MODEL_TYPE} \
            --trt_precision=${trt_precision} \
            --trt_min_subgraph_size=${trt_min_subgraph_size} \
            --use_trt=${use_trt} >> ${log_file} 2>&1 | python3.7 ${CASE_ROOT}/py_mem.py "$OUTPUT_BIN/${exe_bin}" >> ${log_file} 2>&1

        printf "finish ${RED} ${model_name} generate calib, use_trt: ${use_trt}, trt_precision: ${trt_precision}, batch_size: ${batch_size}${NC}\n"
        echo " "
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
        gen_int8_calib "clas_benchmark" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/params
    
        test_int8 "clas_benchmark" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/params
    done

    if [ "${MODEL_TYPE}" == "static" ] || [ "${MODEL_TYPE}" == "static_prune_op" ]; then
        # ssdlite_mobilenet_v3_large
        model_case="ssdlite_mobilenet_v3_large"
        gen_int8_calib "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__model__" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__params__" \
                "3,320,320"

        test_int8 "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__model__" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__params__" \
                "3,320,320"
        
        # ssd_mobilenet_v1_voc
        model_case="ssd_mobilenet_v1_voc"
        gen_int8_calib "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__model__" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__params__" \
                "3,300,300"

        test_int8 "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__model__" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__params__" \
                "3,300,300"
        
        seg_model="deeplabv3p \
                fastscnn \
                hrnet \
                icnet \
                pspnet \
                unet"

        for tests in ${seg_model}
        do
            gen_int8_calib "clas_benchmark" "${tests}" \
                    ${DATA_ROOT}/PaddleSeg/infer_static/${tests}/__model__ \
                    ${DATA_ROOT}/PaddleSeg/infer_static/${tests}/__params__ \
                    "3,512,512"
        
            test_int8 "clas_benchmark" "${tests}" \
                    ${DATA_ROOT}/PaddleSeg/infer_static/${tests}/__model__ \
                    ${DATA_ROOT}/PaddleSeg/infer_static/${tests}/__params__ \
                    "3,512,512"
        done

        # ch_ppocr_mobile_v1.1_cls_infer
        model_case="ch_ppocr_mobile_v1.1_cls_infer"
        gen_int8_calib "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/model" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/params" \
                "3,48,192"

        test_int8 "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/model" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/params" \
                "3,48,192"
        
        # ch_ppocr_mobile_v1.1_det_infer
        model_case="ch_ppocr_mobile_v1.1_det_infer"
        gen_int8_calib "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/model" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/params" \
                "3,640,640"

        test_int8 "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/model" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/params" \
                "3,640,640"
        
        # ch_ppocr_mobile_v1.1_rec_infer
        model_case="ch_ppocr_mobile_v1.1_rec_infer"
        gen_int8_calib "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/model" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/params" \
                "3,32,320"

        test_int8 "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/model" \
                "${DATA_ROOT}/PaddleOCR/${model_case}/params" \
                "3,32,320" "10"
    fi

    printf "${YELLOW} ==== finish benchmark ==== ${NC} \n"
}

model_root=${DATA_ROOT}/PaddleClas/infer_static
if [ $# -ge 1 ]; then
    model_root=$1
fi

main ${model_root}
