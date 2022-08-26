
DOCKER_IMAGE="registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82"
DOCKER_NAME="tipc_prepare_paddlewhl"

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

docker rm -f ${DOCKER_NAME} || echo "remove docker ""${DOCKER_NAME}"" failed"
nvidia-docker run -i --rm \
                  --name ${DOCKER_NAME} \
                  --privileged \
                  --net=host \
                  -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi ${CUDA_SO} ${DEVICES} \
                  -v $(pwd):/workspace \
                  -w /workspace \
                  -u root \
                  -e "HTTP_PROXY=${HTTP_PROXY}" \
                  -e "HTTPS_PROXY=${HTTPS_PROXY}" \
                  -e "no_proxy=${no_proxy:-baidu.com,bcebos.com}" \
                  ${DOCKER_IMAGE} \
                  /bin/bash -c -x "

unset http_proxy
unset https_proxy

python2 -m pip install --retries 10 pycrypto

wget -q --no-proxy  https://paddle-docker-tar.bj.bcebos.com/home/bce_whl.tar.gz --no-check-certificate
tar -zxf bce_whl.tar.gz
wget -q --no-proxy https://paddle-qa.bj.bcebos.com/develop-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-0.0.0.post101-cp37-cp37m-linux_x86_64.whl
python2 bce-python-sdk-0.8.27/BosClient.py paddlepaddle_gpu-0.0.0.post101-cp37-cp37m-linux_x86_64.whl paddle-qa/fullchain_test/whl

"
