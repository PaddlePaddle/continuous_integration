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

    for batch_size in "1"
    do
        echo " "
        printf "start ${YELLOW} ${model_name}, use_gpu: ${use_gpu}, batch_size: ${batch_size}${NC}\n"

        log_file="${LOG_ROOT}/${model_name}_gpu_bz${batch_size}_infer.log"
        if [ "${MODEL_TYPE}" == "static_prune_op" ]; then
            echo "========================== start prune model op attribute +++++++++++++++++++++++++++"
            python ${UTILS_ROOT}/model_clip.py --model_file="${model_path}" \
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
            --use_gpu=${use_gpu} >> ${log_file} 2>&1 | python ${CASE_ROOT}/py_mem.py "$OUTPUT_BIN/${exe_bin}" >> ${log_file} 2>&1

        printf "finish ${RED} ${model_name}, use_gpu: ${use_gpu}, batch_size: ${batch_size}${NC}\n"
        echo " "
    done                               
}

main(){
    printf "${YELLOW} ==== start benchmark ==== ${NC} \n"
    model_root=$1

    class_model="AlexNet"
    
    for tests in ${class_model}
    do
        test_gpu "clas_benchmark" "${tests}" \
                 ${model_root}/${tests}/__model__ \
                 ${model_root}/${tests}/params
    done

    if [ "${MODEL_TYPE}" == "static" ] || [ "${MODEL_TYPE}" == "static_prune_op" ]; then
        # ssdlite_mobilenet_v3_large
        model_case="ssdlite_mobilenet_v3_large"
        test_gpu "clas_benchmark" "${model_case}" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__model__" \
                "${DATA_ROOT}/PaddleDetection/infer_static/${model_case}/__params__" \
                "3,320,320"
    fi

    printf "${YELLOW} ==== finish benchmark ==== ${NC} \n"
}

model_root=${DATA_ROOT}/PaddleClas/infer_static
if [ $# -ge 1 ]; then
    model_root=$1
fi

main ${model_root}
