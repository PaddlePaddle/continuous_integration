#! /bin/bash

set -ex

#repo_list="PaddleOCR PaddleClas PaddleSeg PaddleNLP PaddleDetection PaddleRec DeepSpeech"

REPO=$1
DOCKER_IMAGE=registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82
DOCKER_NAME=paddle_whole_chain_test
#COMPILE_PATH=https://paddle-qa.bj.bcebos.com/paddle-pipeline/Master_GpuAll_LinuxUbuntu_Gcc82_Cuda10.1_Trton_Py37_Compile_H_DISTRIBUTE/latest/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl 
COMPILE_PATH=https://paddle-qa.bj.bcebos.com/paddle-pipeline/Debug_GpuAll_LinuxUbuntu_Gcc82_Cuda10.1_Trton_Py37_Compile_H_DISTRIBUTE_Release/latest/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl

# define version compare function
function version_lt() { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" != "$1"; }


# start docker and run test cases
docker pull ${DOCKER_IMAGE}

# check nvidia-docker version
nv_docker_version=`nvidia-docker version | grep NVIDIA | cut -d" " -f3`
if version_lt ${nv_docker_version} 2.0.0; then
   echo -e "nv docker version ${nv_docker_version} is less than 2.0.0, should map CUDA_SO and DEVICES to docker"
   export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
   export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
fi


export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}


docker rm -f ${DOCKER_NAME} || echo "remove docker paddle_whole_chain_test failed"
nvidia-docker run -i --rm \
                  --name ${DOCKER_NAME} \
                  --privileged \
                  --net=host \
                  --shm-size=128G \
                  -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi ${CUDA_SO} ${DEVICES} \
                  -v $(pwd):/workspace \
                  -w /workspace \
                  -u root \
                  -e "FLAGS_fraction_of_gpu_memory_to_use=0.01" \
                  -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
                  -e "TIPC_MODE=${TIPC_MODE}" \
                  -e "http_proxy=${http_proxy}" \
                  -e "https_proxy=${https_proxy}" \
                  ${DOCKER_IMAGE} \
                  /bin/bash -c -x "
export PATH=/home/cmake-3.16.0-Linux-x86_64/bin:/workspace/run_env:/usr/local/gcc-8.2/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
if [[ $TIPC_MODE == "cpp_infer" ]]; then
    cd ./${REPO}
    cp ../continuous_integration/tipc/tipc_run_cpp.sh .
    export http_proxy
    export https_proxy
    sh tipc_run_cpp.sh
    exit $?
fi
"
