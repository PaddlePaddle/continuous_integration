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

    rcnn_model="mask_rcnn_r50_1x \
                faster_rcnn_r50_1x \
                faster_rcnn_dcn_r50_vd_fpn_3x_server_side"
                
    for tests in ${rcnn_model}
    do
        gen_int8_calib "rcnn_benchmark" "${tests}" \
                ${model_root}/${tests}/__model__ \
                ${model_root}/${tests}/__params__ \
                "3,640,640"

        test_int8 "rcnn_benchmark" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/__params__ \
                 "3,640,640" "40"
    done

    yolo_model="ppyolo_mobilenet_v3_large \
                yolov3_darknet \
                yolov3_mobilenet_v3 \
                yolov3_r50vd_dcn \
                yolov4_cspdarknet"
                
    for tests in ${yolo_model}
    do
        gen_int8_calib "yolo_benchmark" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/__params__ \
                 "3,608,608"
    
        test_int8 "yolo_benchmark" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/__params__ \
                 "3,608,608"
    done

    printf "${YELLOW} ==== finish benchmark ==== ${NC} \n"
}

model_root=${DATA_ROOT}/PaddleDetection/infer_static
if [ $# -ge 1 ]; then
    model_root=$1
fi

main ${model_root}
