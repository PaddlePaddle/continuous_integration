#! /bin/bash

set -ex

REPO=$1
CHAIN=$2
PADDLE_WHL=$3
DOCKER_IMAGE=$4
CODE_BOS=$5
FRAME_BRANCH=$6
SENDER=$7
RECVIER=$8
MAIL_PROXY=$9
DOCKER_NAME=${DOCKER_NAME:-paddle_tipc_test_${REPO}_${CHAIN}}
PADDLE_INFERENCE_TGZ=${PADDLE_INFERENCE_TGZ:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Master_GpuAll_LinuxCentos_Gcc82_Cuda10.1_cudnn7.6_trt6015_onort_Py38_Compile_H/latest/paddle_inference.tgz}
#PADDLE_INFERENCE_TGZ=${PADDLE_INFERENCE_TGZ:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-Centos-Gcc82-Cuda102-Cudnn76-Trt6018-Py38-Compile/latest/paddle_inference.tgz}
BCE_CLIENT_PATH=${BCE_CLIENT_PATH:-/home/work/bce-client}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
DEBUG=${DEBUG:-False}
TF=${TF:-False}

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
                  --shm-size=128G \
                  -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi ${CUDA_SO} ${DEVICES} \
                  -v $(pwd):/workspace \
                  -v /home/work/bce-client:${BCE_CLIENT_PATH} \
                  -w /workspace \
                  -u root \
                  -e "FLAGS_fraction_of_gpu_memory_to_use=0.01" \
                  -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
                  -e "TIMEOUT=${TIMEOUT}" \
                  -e "PADDLE_INFERENCE_TGZ=${PADDLE_INFERENCE_TGZ}" \
                  -e "HTTP_PROXY=${HTTP_PROXY}" \
                  -e "HTTPS_PROXY=${HTTPS_PROXY}" \
                  -e "grep_v_models=${grep_v_models}" \
                  -e "grep_models=${grep_models}" \
                  -e "TF=${TF}" \
                  -e "DEBUG=${DEBUG:-False}" \
                  -e "no_proxy=${no_proxy:-baidu.com,bcebos.com}" \
                  ${DOCKER_IMAGE} \
                  /bin/bash -c -x "

unset http_proxy
unset https_proxy

apt-get update
apt-get install apt-transport-https
wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub | apt-key add -

mkdir -p run_env
ln -s /usr/local/bin/python3.7 run_env/python
ln -s /usr/local/bin/pip3.7 run_env/pip
export PATH=/home/cmake-3.16.0-Linux-x86_64/bin:/workspace/run_env:/usr/local/gcc-8.2/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export REPO=$REPO
export CHAIN=$CHAIN
export DEBUG=${DEBUG:-False}
export TF=${TF:-False}

export http_proxy=
export https_proxy=
echo $http_proxy $https_proxy

python -m pip install --retries 50 --upgrade pip -i https://mirror.baidu.com/pypi/simple
python -m pip config set global.index-url https://mirror.baidu.com/pypi/simple;
python -m pip install pymysql
wget -q --no-proxy ${PADDLE_WHL}
python -m pip install ./\`basename ${PADDLE_WHL}\`

if [[ ${CHAIN} == "chain_distribution" ]]
then
    cd ${REPO}
    ## 拉取pdc安装包 , 安装pdc client
    wget -q --no-proxy https://paddle-qa.bj.bcebos.com/fullchain_test/tools/paddlecloud-cli.tar.gz
    tar -zxf paddlecloud-cli.tar.gz
    cd paddlecloud-cli
    python setup.py install
    cd -
    cat pdc_conf.ini > ~/.paddlecli/config 
    bash tipc_run.sh ${REPO} ${CHAIN} ${PADDLE_WHL} ${FRAME_BRANCH} ${DOCKER_IMAGE} ${CODE_BOS} ${SENDER} ${RECVIER} ${MAIL_PROXY}
else
cd ./AutoLog
python -m pip install --retries 10 -r requirements.txt
python setup.py bdist_wheel
cd -
python -m pip install ./AutoLog/dist/*.whl


cd ./${REPO}
REPO_PATH=\`pwd\`
if [[ $REPO == "PaddleNLP" ]]; then
    cd tests
fi
if [[ $REPO == "PaddleGAN" ]]; then
    python -m pip install -v -e . #安装ppgan
fi
if [[ $REPO == "PaddleRec" ]]; then
    python -m pip install pgl
    python -m pip install h5py
    python -m pip install nltk
fi
if [[ $REPO == "PARL" ]]; then
    pip uninstall protobuf -y
    pip install protobuf=3.19.0
    export http_proxy=${HTTP_PROXY}
    export https_proxy=${HTTPS_PROXY}
    pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
    export http_proxy=
    export https_proxy=
fi
python2 -m pip install --retries 10 pycrypto
python -m pip install --retries 10 Cython
python -m pip install --retries 10 distro
python -m pip install --retries 10 opencv-python
python -m pip install --retries 10 wget
python -m pip install --retries 10 pynvml
python -m pip install --retries 10 cup
python -m pip install --retries 10 pandas
python -m pip install --retries 10 openpyxl
python -m pip install --retries 10 psutil
python -m pip install --retries 10 GPUtil
python -m pip install --retries 10 paddleslim
#python -m pip install --retries 10 paddlenlp
python -m pip install --retries 10 attrdict
python -m pip install --retries 10 pyyaml
python -m pip install --retries 10 visualdl 
python -c 'from visualdl import LogWriter'
#git clone -b develop https://github.com/PaddlePaddle/PaddleSlim.git
#cd PaddleSlim     
#python -m pip install -r requirements.txt
#python setup.py install
#cd ..
python -m pip install --retries 10 -r requirements.txt

if [[ $REPO == "PaddleSeg" ]]; then
    pip install -e .
    python -m pip install --retries 50 scikit-image
    python -m pip install numba
    python -m pip install sklearn
    python -m pip install pymatting
    if [[ $CHAIN == "chain_serving_cpp" ]]; then
        pip install SimpleITK -i https://pypi.tuna.tsinghua.edu.cn/simple
        pip install decord==0.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
        pip install av==8.0.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi
fi
if [[ $REPO == "PaddleNLP" ]]; then
    python -m pip install --retries 10 paddlenlp
fi
if [[ $REPO == "PaddleOCR" ]] && [[ $CHAIN == "chain_pact_infer_python" ]]; then
    python -m pip install --retries 10 paddlenlp
fi
if [[ $REPO == "PaddleVideo" ]]; then
    python -m pip install --retries 10 paddlenlp
    python -m pip install --retries 10 SimpleITK
    python -m pip install --retries 10 lmdb 
fi
if [[ $REPO == "PaddleClas" ]]; then
    python -m pip install --retries 10 paddleclas
fi
cp \$REPO_PATH/../continuous_integration/tipc/tipc_run.sh .
cp \$REPO_PATH/../continuous_integration/tipc/upload.sh .
cp -r \$REPO_PATH/../continuous_integration/tipc/configs .
cp -r \$REPO_PATH/../continuous_integration/tipc/model_list.py .
cp \$REPO_PATH/../continuous_integration/tipc/test_export_shell.sh .
cp \$REPO_PATH/../continuous_integration/tipc/writedb.py .
cp \$REPO_PATH/../continuous_integration/tipc/upload_log.sh .


bash -x tipc_run.sh ${REPO} ${CHAIN} ${PADDLE_WHL} ${FRAME_BRANCH} ${DOCKER_IMAGE} ${CODE_BOS} ${SENDER} ${RECVIER} ${MAIL_PROXY}

fi

"


