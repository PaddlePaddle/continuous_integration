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
    use_gpu=true;

    for batch_size in "1" "2" "4"
    do
        echo " "
        printf "start ${YELLOW} ${model_name}, use_gpu: ${use_gpu}, batch_size: ${batch_size}${NC}\n"

        log_file="${LOG_ROOT}/${model_name}_gpu_bz${batch_size}_infer.log"
        if [ "${MODEL_TYPE}" == "static_prune_op" ]; then
            echo "========================== start prune model op attribute +++++++++++++++++++++++++++"
            python3.7 ${UTILS_ROOT}/model_clip.py --model_file="${model_path}" \
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
            --use_gpu=${use_gpu} >> ${log_file} 2>&1 | python3.7 ${CASE_ROOT}/py_mem.py "$OUTPUT_BIN/${exe_bin}" >> ${log_file} 2>&1

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

    trt_min_subgraph_size="6"
    if [ $# -ge 6 ]; then
        trt_min_subgraph_size=$6
    fi

    printf "${YELLOW} ${model_name} input image_shape = ${image_shape} ${NC} \n";
    use_gpu=true;
    use_trt=true;

    # Tesla T4 can run fp16
    if [ "$gpu_type" == "T4" ]; then
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

            log_file="${LOG_ROOT}/${model_name}_trt_${trt_precision}_bz${batch_size}_infer.log"
            if [ "${MODEL_TYPE}" == "static_prune_op" ]; then
                echo "========================== start prune model op attribute +++++++++++++++++++++++++++"
                python3.7 ${UTILS_ROOT}/model_clip.py --model_file="${model_path}" \
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
                --trt_min_subgraph_size=${trt_min_subgraph_size} \
                --trt_precision=${trt_precision} \
                -use_trt=${use_trt} >> ${log_file} 2>&1 | python3.7 ${CASE_ROOT}/py_mem.py "$OUTPUT_BIN/${exe_bin}" >> ${log_file} 2>&1

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
        test_gpu "rcnn_benchmark" "${tests}" \
                ${model_root}/${tests}/__model__ \
                ${model_root}/${tests}/__params__ \
                "3,640,640"

        test_trt "rcnn_benchmark" "${tests}" \
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
        test_gpu "yolo_benchmark" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/__params__ \
                 "3,608,608"
    
        test_trt "yolo_benchmark" "${tests}" \
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
