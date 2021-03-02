#!/bin/bash
export RED='\033[0;31m' # red color
export NC='\033[0m' # no color
export YELLOW='\033[33m' # yellow color

test_gpu(){
    exe_bin=$1 # ./clas_benchmark.py
    model_name=$2
    model_path=$3

    image_shape="3,224,224"
    if [ $# -ge 4 ]; then
        image_shape=$4
    fi

    input_node="import/inputs:0"
    if [ $# -ge 5 ]; then
        input_node=$5
    fi

    output_node="resnet_v1_101/predictions/Reshape_1:0"
    if [ $# -ge 6 ]; then
        output_node=$6
    fi

    repeats=1000
    if [ $# -ge 7 ]; then
        repeats=$7
    fi    

    printf "${YELLOW} ${model_name} input image_shape = ${image_shape} ${NC} \n";
    use_gpu=true;

    for batch_size in "1" "2" "4"
    do
        echo " "
        printf "start ${YELLOW} ${model_name}, use_gpu: ${use_gpu}, batch_size: ${batch_size}${NC}\n"

        log_file="${LOG_ROOT}/${model_name}_gpu_bz${batch_size}_infer.log"
        python3 clas_benchmark.py --model_name=${model_name} \
                                  --batch_size=${batch_size} \
                                  --model_type=static \
                                  --image_shape=${image_shape} \
                                  --model_path=${model_path} \
                                  --input_node=${input_node} \
                                  --output_node=${output_node} \
                                  --repeats=${repeats} \
                                  --use_gpu > ${log_file} 2>&1 
        printf "finish ${RED} ${model_name}, use_gpu: ${use_gpu}, batch_size: ${batch_size}${NC}\n"
        echo " "
    done                               
}

test_trt(){
    exe_bin=$1 # ./clas_benchmark.py
    model_name=$2
    model_path=$3

    image_shape="3,224,224"
    if [ $# -ge 4 ]; then
        image_shape=$4
    fi

    input_node="import/inputs:0"
    if [ $# -ge 5 ]; then
        input_node=$5
    fi

    output_node="resnet_v1_101/predictions/Reshape_1:0"
    if [ $# -ge 6 ]; then
        output_node=$6
    fi

    repeats=1000
    if [ $# -ge 7 ]; then
        repeats=$7
    fi    

    printf "${YELLOW} ${model_name} input image_shape = ${image_shape} ${NC} \n";
    use_gpu=true;
    use_trt=true;

    for batch_size in "1" "2" "4"
    do
        echo " "
        printf "start ${YELLOW} ${model_name}, use_trt: ${use_trt}, batch_size: ${batch_size}${NC}\n"

        log_file="${LOG_ROOT}/${model_name}_trt_bz${batch_size}_infer.log"
        python3 clas_benchmark.py --model_name=${model_name} \
                                  --batch_size=${batch_size} \
                                  --model_type=static \
                                  --image_shape=${image_shape} \
                                  --model_path=${model_path} \
                                  --input_node=${input_node} \
                                  --output_node=${output_node} \
                                  --repeats=${repeats} \
                                  --use_trt \
                                  --use_gpu > ${log_file} 2>&1 
        printf "finish ${RED} ${model_name}, use_trt: ${use_trt}, batch_size: ${batch_size}${NC}\n"
        echo " "
    done                               
}

test_gpu_xla(){
    exe_bin=$1 # ./clas_benchmark.py
    model_name=$2
    model_path=$3

    image_shape="3,224,224"
    if [ $# -ge 4 ]; then
        image_shape=$4
    fi

    input_node="import/inputs:0"
    if [ $# -ge 5 ]; then
        input_node=$5
    fi

    output_node="resnet_v1_101/predictions/Reshape_1:0"
    if [ $# -ge 6 ]; then
        output_node=$6
    fi

    repeats=1000
    if [ $# -ge 7 ]; then
        repeats=$7
    fi    

    printf "${YELLOW} ${model_name} input image_shape = ${image_shape} ${NC} \n";
    use_gpu=true;
    use_xla=true;

    for batch_size in "1" "2" "4"
    do
        echo " "
        printf "start ${YELLOW} ${model_name}, use_xla: ${use_xla}, batch_size: ${batch_size}${NC}\n"

        log_file="${LOG_ROOT}/${model_name}_xla_bz${batch_size}_infer.log"
        python3 clas_benchmark.py --model_name=${model_name} \
                                  --batch_size=${batch_size} \
                                  --model_type=static \
                                  --image_shape=${image_shape} \
                                  --model_path=${model_path} \
                                  --input_node=${input_node} \
                                  --output_node=${output_node} \
                                  --repeats=${repeats} \
                                  --use_xla \
                                  --use_gpu > ${log_file} 2>&1 
        printf "finish ${RED} ${model_name}, use_xla: ${use_xla}, batch_size: ${batch_size}${NC}\n"
        echo " "
    done                               
}


resnet101(){
    model_root=$1

    test_gpu "./clas_benchmark.py" "resnet_v1_101" \
             "${model_root}/resnet101-tf/resnet_v1_101.pb" \
             "3,224,224" "import/inputs:0" \
             "resnet_v1_101/predictions/Reshape_1:0"
    
    test_trt "./clas_benchmark.py" "resnet_v1_101" \
             "${model_root}/resnet101-tf/resnet_v1_101.pb" \
             "3,224,224" "import/inputs:0" \
             "resnet_v1_101/predictions/Reshape_1:0"
    
    test_gpu_xla "./clas_benchmark.py" "resnet_v1_101" \
             "${model_root}/resnet101-tf/resnet_v1_101.pb" \
             "3,224,224" "import/inputs:0" \
             "resnet_v1_101/predictions/Reshape_1:0"
}

MobilenetV2(){
    model_root=$1
    test_gpu "./clas_benchmark.py" "MobilenetV2" \
            "${model_root}/mobilenet_v2-tf/mobilenet_v2_1.0_224_frozen.pb" \
            "3,224,224" "import/input:0" \
            "MobilenetV2/Predictions/Reshape_1:0"

    test_trt "./clas_benchmark.py" "MobilenetV2" \
            "${model_root}/mobilenet_v2-tf/mobilenet_v2_1.0_224_frozen.pb" \
            "3,224,224" "import/input:0" \
            "MobilenetV2/Predictions/Reshape_1:0"

    test_gpu_xla "./clas_benchmark.py" "MobilenetV2" \
            "${model_root}/mobilenet_v2-tf/mobilenet_v2_1.0_224_frozen.pb" \
            "3,224,224" "import/input:0" \
            "MobilenetV2/Predictions/Reshape_1:0"
}

ssd_mobilenet_v1(){
    model_root=$1
    test_gpu "./clas_benchmark.py" "ssd_mobilenet_v1" \
             "${model_root}/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb" \
             "3,300,300" "import/image_tensor:0" \
             "detection_classes:0"

    test_trt "./clas_benchmark.py" "ssd_mobilenet_v1" \
             "${model_root}/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb" \
             "3,300,300" "import/image_tensor:0" \
             "detection_classes:0"

    test_gpu_xla "./clas_benchmark.py" "ssd_mobilenet_v1" \
             "${model_root}/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb" \
             "3,300,300" "import/image_tensor:0" \
             "detection_classes:0"
}

deeplabv3p(){
    test_gpu "./clas_benchmark.py" "deeplabv3p" \
             "${model_root}/deeplabv3p-tf/frozen_inference_graph.pb" \
             "3,224,224" "import/ImageTensor:0" \
             "SemanticPredictions:0" "100"

    # out of memory
    # test_gpu_xla "./clas_benchmark.py" "deeplabv3p" \
    #          "${model_root}/deeplabv3p-tf/frozen_inference_graph.pb" \
    #          "3,224,224" "import/ImageTensor:0" \
    #          "SemanticPredictions:0" "100"
}

unet(){
    test_gpu "./clas_benchmark.py" "unet-tf" \
             "${model_root}/unet-tf/unet.pb" \
             "1,512,512" "import/Placeholder:0" \
             "UNet/conv2d_23/BiasAdd:0"

    test_trt "./clas_benchmark.py" "unet-tf" \
             "${model_root}/unet-tf/unet.pb" \
             "1,512,512" "import/Placeholder:0" \
             "UNet/conv2d_23/BiasAdd:0"

    test_gpu_xla "./clas_benchmark.py" "unet-tf" \
             "${model_root}/unet-tf/unet.pb" \
             "1,512,512" "import/Placeholder:0" \
             "UNet/conv2d_23/BiasAdd:0"
}
    
faster_rcnn(){
    test_gpu "./clas_benchmark.py" "faster_rcnn" \
             "${model_root}/faster_rcnn_resnet50_coco-tf/frozen_inference_graph.pb" \
             "3,640,640" "import/image_tensor:0" \
             "detection_classes:0" "500"
    
    test_trt "./clas_benchmark.py" "faster_rcnn" \
             "${model_root}/faster_rcnn_resnet50_coco-tf/frozen_inference_graph.pb" \
             "3,640,640" "import/image_tensor:0" \
             "detection_classes:0" "500"

    test_gpu_xla "./clas_benchmark.py" "faster_rcnn" \
             "${model_root}/faster_rcnn_resnet50_coco-tf/frozen_inference_graph.pb" \
             "3,640,640" "import/image_tensor:0" \
             "detection_classes:0" "500"
}

main(){
    printf "${YELLOW} ==== start benchmark ==== ${NC} \n"
    model_root=$1

    resnet101 $1
    
    MobilenetV2 $1

    ssd_mobilenet_v1 $1

    deeplabv3p $1

    unet $1
    
    faster_rcnn $1 
    printf "${YELLOW} ==== finish benchmark ==== ${NC} \n"
}

LOG_ROOT="./log"
mkdir ${LOG_ROOT}
model_root="./Data/TensorFlow"

if [ $# -ge 1 ]; then
    model_root=$1
fi

main ${model_root}
