#!/bin/bash
export RED='\033[0;31m' # red color
export NC='\033[0m' # no color
export YELLOW='\033[33m' # yellow color

function test_det_cpu(){
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
            --model_type=${MODEL_TYPE} \
            --repeats=10 \
            --use_gpu=${use_gpu} >> ${log_file} 2>&1 | python3.7 ${CASE_ROOT}/py_mem.py "$OUTPUT_BIN/${exe_bin}" >> ${log_file} 2>&1

        printf "finish ${RED} ${model_name}, use_gpu: ${use_gpu}, batch_size: ${batch_size}${NC}\n"
        echo " "
    done                               
}


function test_det_mkldnn(){
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
                --repeats=10 \
                --model_type=${MODEL_TYPE} \
                --cpu_math_library_num_threads=${cpu_math_library_num_threads} \
                --use_mkldnn_=${use_mkldnn} >> ${log_file} 2>&1 | python3.7 ${CASE_ROOT}/py_mem.py "$OUTPUT_BIN/${exe_bin}" >> ${log_file} 2>&1

            printf "finish ${RED} ${model_name}, use_mkldnn: ${use_mkldnn}, cpu_math_library_num_threads: ${cpu_math_library_num_threads}, batch_size: ${batch_size}${NC}\n"
            echo " "
        done
    done                               
}

function run_det_mkl_func(){
    model_root=${DATA_ROOT}/PaddleDetection/infer_static
    if [ $# -ge 1 ]; then
        model_root=$1
    fi
    printf "${YELLOW} ==== start benchmark ==== ${NC} \n"
    declare name_batch_size=$2[@]
    cpu_batch_size=(${!name_batch_size})
    declare name_threads=$3[@]
    cpu_num_threads=(${!name_threads})

    rcnn_model="mask_rcnn_r50_1x \
                faster_rcnn_r50_1x \
                faster_rcnn_dcn_r50_vd_fpn_3x_server_side"
                
    for tests in ${rcnn_model}
    do
        test_det_cpu "rcnn_benchmark" "${tests}" \
                ${model_root}/${tests}/__model__ \
                ${model_root}/${tests}/__params__ \
                cpu_batch_size "3,640,640"

        test_det_mkldnn "rcnn_benchmark" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/__params__ \
                 cpu_batch_size cpu_num_threads "3,640,640" "40"
    done

    yolo_model="ppyolo_mobilenet_v3_large \
                yolov3_darknet \
                yolov3_mobilenet_v3 \
                yolov3_r50vd_dcn \
                yolov4_cspdarknet"
                
    for tests in ${yolo_model}
    do
        test_det_cpu "yolo_benchmark" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/__params__ \
                 cpu_batch_size "3,608,608"
    
        test_det_mkldnn "yolo_benchmark" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/__params__ \
                 cpu_batch_size cpu_num_threads "3,608,608"
    done

    printf "${YELLOW} ==== finish benchmark ==== ${NC} \n"
}
