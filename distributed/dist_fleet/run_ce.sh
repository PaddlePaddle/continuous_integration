#!/bin/bash

if [[ -t 1 ]]
then
    YELLOW="$( echo -e "\e[33m" )"
    GREEN="$( echo -e "\e[32m" )"
    RED="$( echo -e "\e[31m" )"
    NORMAL="$( echo -e "\e[0m" )"
fi

function _yellow(){ echo "$YELLOW""$@""$NORMAL";}
function _green(){ echo "$GREEN""$@""$NORMAL";}
function _red(){ echo "$RED""$@""$NORMAL";}

function _message(){ echo "$@" >&2;}
function _warn(){ echo $(_yellow '==> WARN:') "$@" >&2;}
function _info(){ echo $(_green '==> ') "$@" >&2;}
function _error(){ echo $(_red '==> ERROR:') "$@" >&2;}
function _fatal(){ echo $(_red '==> ERROR:') "$@" >&2; exit 1;}


function _print_usage(){
    _message "usage: $0 [options]"
    _message "    -h  optional   Print this help message"
    _message "    -b  dir of cts_work_path"
    _message "    -g  device_type  GPU | CPU"
    _message "    -c  cuda_version 9.0|10.0"
    _message "    -n  cudnn_version 7"
    _message "    -a  image_branch develop | 1.6 | pr"
    _message "    -r  run_frequency True | False"
    _message "    -p  all_path contains dir of images such as /ssd1/ljh"
    _message "    -w  switch of pslib ON | OFF (default is OFF)"
    _message "    -e  benchmark alarm email address"
}


function _init_parameters(){
    cts_work_path=$(pwd)
    all_path=/ssd1/ljh
    device_type='GPU'
    cuda_version=9.0
    cudnn_version=7
    image_branch='develop'
    with_pslib='OFF'
    with_mkl="ON"
    run_frequency="DAILY"
    email_address=liangjinhua01@baidu.com
}

_init_parameters

while [[ $# -gt 0 ]]; do
    case "$1" in
    -h|--help) _print_usage; exit 0 ;;
    -b|--cts_work_path)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        cts_work_path=$2
        shift; shift
        ;;
    -g|--device_type)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        device_type=$2
        shift; shift
        ;;
    -c|--cuda_version)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        cuda_version=$2
        shift; shift
        ;;
    -n|--cudnn_version)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        cudnn_version=$2
        shift; shift
        ;;
    -a|--image_branch)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        image_branch=$2
        shift; shift
        ;;
    -r|--run_frequency)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        run_frequency=$2
        shift; shift
        ;;
    -p|--all_path)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        all_path=$2
        shift; shift
        ;;
    -w|--with_pslib)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        with_pslib=$2
        shift; shift
        ;;
    -e|--email_address)
        [[ "X$2" == "X" ]] && _fatal "option $1 needs an argument!"
        email_address=$2
        shift; shift
        ;;
    *)
        _print_usage
        _fatal "Unrecongnized option $1"
        ;;
   esac
done

my_http_proxy=http://172.19.57.45:3128

export PADDLE_REPO="https://github.com/PaddlePaddle/Paddle.git"

#build paddle
function build(){
    cd ${cts_work_path}/Paddle
    rm -rf build
    image_commit_id=$(git log|head -n1|awk '{print $2}')
    _info "image_commit_id is: "${image_commit_id}

    PADDLE_DEV_NAME=docker.io/paddlepaddle/paddle_manylinux_devel:cuda${cuda_version}_cudnn${cudnn_version}
    version=`date -d @$(git log -1 --pretty=format:%ct) "+%Y.%m%d.%H%M%S"`
    image_branch=$(echo ${image_branch} | rev | cut -d'/' -f 1 | rev)
    if [[ ${device_type} == 'cpu' || ${device_type} == "CPU" ]]; then
        PADDLE_VERSION=${version}".${image_branch//-/_}"
        IMAGE_NAME=paddlepaddle-0.0.0.${PADDLE_VERSION}-cp27-cp27mu-linux_x86_64.whl
        with_gpu="OFF"
        if [[ ${with_pslib} == 'ON' ]]; then
            PADDLE_VERSION=${PADDLE_VERSION}.pslib
            IMAGE_NAME=paddlepaddle-0.0.0.${PADDLE_VERSION}-cp27-cp27mu-linux_x86_64.whl
            with_mkl="OFF"
            PADDLE_DEV_NAME=hub.baidubce.com/paddlepaddle/paddle:latest-dev
        fi
    else
        PADDLE_VERSION=${version}'.post'$(echo ${cuda_version}|cut -d "." -f1)${cudnn_version}".${image_branch//-/_}"
        IMAGE_NAME=paddlepaddle_gpu-0.0.0.${PADDLE_VERSION}-cp27-cp27mu-linux_x86_64.whl
        with_gpu='ON'
    fi

    _info "image_name is: "${IMAGE_NAME}
    docker pull ${PADDLE_DEV_NAME}

    #double check1: In some case, docker would hang while compiling paddle, so to avoid re-compilem, need this
    if [[ -e ${all_path}/images/${IMAGE_NAME} ]]
    then
        echo "image had built, begin running models"
        return
    else
        echo "image not found, begin building"
    fi

    if [[ ${with_pslib} == 'ON' ]]; then
        docker run -i --rm -v $PWD:/paddle \
            -w /paddle \
            -e "http_proxy=${my_http_proxy}" \
            -e "https_proxy=${my_http_proxy}" \
            ${PADDLE_DEV_NAME} \
            /bin/bash -c "mkdir -p /paddle/build && cd /paddle/build; pip install protobuf; \
                         cmake .. -DPY_VERSION=2.7 -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
                           -DWITH_PSLIB=ON -DWITH_PSLIB_BRPC=ON; \
                         make -j$(nproc)"
            build_name="paddlepaddle-0.0.0-cp27-cp27mu-linux_x86_64.whl"
            mv ./build/python/dist/${build_name} ./build/python/dist/${IMAGE_NAME}
    else
        nvidia-docker run -i --rm -v $PWD:/paddle \
            -w /paddle \
            -e "CMAKE_BUILD_TYPE=Release" \
            -e "PYTHON_ABI=cp27-cp27mu" \
            -e "PADDLE_VERSION=0.0.0.${PADDLE_VERSION}" \
            -e "WITH_DOC=OFF" \
            -e "WITH_AVX=ON" \
            -e "WITH_GPU=${with_gpu}" \
            -e "WITH_PSLIB=${with_pslib}" \
            -e "WITH_PSLIB_BRPC=${with_pslib}" \
            -e "WITH_TEST=OFF" \
            -e "RUN_TEST=OFF" \
            -e "WITH_GOLANG=OFF" \
            -e "WITH_SWIG_PY=ON" \
            -e "WITH_PYTHON=ON" \
            -e "WITH_C_API=OFF" \
            -e "WITH_STYLE_CHECK=OFF" \
            -e "WITH_TESTING=OFF" \
            -e "CMAKE_EXPORT_COMPILE_COMMANDS=ON" \
            -e "WITH_MKL=${with_mkl}" \
            -e "BUILD_TYPE=Release" \
            -e "WITH_DISTRIBUTE=ON" \
            -e "WITH_FLUID_ONLY=OFF" \
            -e "CMAKE_VERBOSE_MAKEFILE=OFF" \
            -e "http_proxy=${my_http_proxy}" \
            -e "https_proxy=${my_http_proxy}" \
            ${PADDLE_DEV_NAME} \
             /bin/bash -c "paddle/scripts/paddle_build.sh build"
    fi
    mkdir -p ./output

    if [[ -d ${all_path}/images ]]; then
        _info "images dir already exists"
    else
        mkdir -p ${all_path}/images
    fi

    if [[ -d ${all_path}/logs/log_${PADDLE_VERSION} ]]; then
        _info "local logs dir already exists"
        rm -rf ${all_path}/logs/log_${PADDLE_VERSION}
    else
        mkdir -p ${all_path}/logs/log_${PADDLE_VERSION}
    fi

    build_link="${CE_SERVER}/viewLog.html?buildId=${BUILD_ID}&buildTypeId=${BUILD_TYPE_ID}&tab=buildLog"
    echo "build log link: ${build_link}"

    if [[ -s $(pwd)/build/python/dist/${IMAGE_NAME} ]]; then
        cp ./build/python/dist/${IMAGE_NAME} ${all_path}/images/${IMAGE_NAME}
    else
        echo "${IMAGE_NAME} build failed, it's empty!!"
        sendmail -t ${email_address} <<EOF
From:paddle_benchmark@baidu.com
SUBJECT:benchmark运行结果报警, 请检查
Content-type: text/plain
PADDLE BUILD FAILED!!
详情请点击: ${build_link}
EOF
        exit 1
    fi
}

function run(){
    cd ${cts_work_path}/baidu/paddle/test/cts_test/dist_fleet
    RUN_IMAGE_NAME=paddlepaddle/paddle_manylinux_devel:cuda${cuda_version}_cudnn${cudnn_version}
    nvidia-docker run -i --rm \
        -v /home/work:/home/work \
        -v ${all_path}:${all_path} \
        -v ${cts_work_path}:${cts_work_path} \
        -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi \
        -v /usr/bin/monquery:/usr/bin/monquery \
        --net=host \
        --privileged \
        -w ${cts_work_path}/baidu/paddle/test/cts_test/dist_fleet \
        -e "http_proxy=${my_http_proxy}" \
        -e "https_proxy=${my_http_proxy}" \
        -e "DEVICE_TYPE=${device_type}" \
        -e "PSLIB=${with_pslib}" \
        -e "IMAGE_NAME=${all_path}/images/${IMAGE_NAME}" \
        -e "RUN_FREQUENCY=${run_frequency}" \
        -e "ALL_PATH=${all_path}" \
        -e "LOG_PATH=${all_path}/logs/log_${PADDLE_VERSION}" \
        ${RUN_IMAGE_NAME} \
        /bin/bash -c "bash ci_fleet_run.sh"
}

function send_email(){
    if [[ -e ${all_path}/logs/log_${PADDLE_VERSION}/mail.html ]]; then
        cat ${all_path}/logs/log_${PADDLE_VERSION}/mail.html |sendmail -t ${email_address}
    fi
}

function zip_log(){
    _info $(pwd)
    if [[ -d ${all_path}/logs/log_${PADDLE_VERSION} ]]; then
        rm -rf output/*
        tar -zcvf output/log_${PADDLE_VERSION}.tar.gz ${all_path}/logs/log_${PADDLE_VERSION}
        cp ${all_path}/images/${IMAGE_NAME}  output/
    fi
}

build
run
#zip_log
#send_email
