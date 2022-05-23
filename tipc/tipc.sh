#! /bin/bash

set -ex

REPO=$1
CHAIN=$2
DOCKER_IMAGE=${DOCKER_IMAGE:-registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82}
DOCKER_NAME=${DOCKER_NAME:-paddle_tipc_test_${REPO}_${CHAIN}}
PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Debug_GpuAll_LinuxUbuntu_Gcc82_Cuda10.1_Trton_Py37_Compile_H_DISTRIBUTE_Release/latest/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl}
PADDLE_INFERENCE_TGZ=${PADDLE_INFERENCE_TGZ:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Master_GpuAll_LinuxCentos_Gcc82_Cuda10.1_cudnn7.6_trt6015_Py38_Compile_H/latest/paddle_inference.tgz}
BCE_CLIENT_PATH=${BCE_CLIENT_PATH:-/home/work/bce-client}

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
                  -e "no_proxy=${no_proxy:-baidu.com,bcebos.com}" \
                  ${DOCKER_IMAGE} \
                  /bin/bash -c -x "
unset http_proxy
unset https_proxy

apt-get install apt-transport-https
wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub | apt-key add -

mkdir -p run_env
ln -s /usr/local/bin/python3.7 run_env/python
ln -s /usr/local/bin/pip3.7 run_env/pip
export PATH=/home/cmake-3.16.0-Linux-x86_64/bin:/workspace/run_env:/usr/local/gcc-8.2/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export REPO=$REPO
export CHAIN=$CHAIN

unset http_proxy
unset https_proxy
echo $http_proxy $https_proxy
python -m pip install --retries 50 --upgrade pip -i https://mirror.baidu.com/pypi/simple
python -m pip config set global.index-url https://mirror.baidu.com/pypi/simple;
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
python -m pip install --retries 10 -r requirements.txt
wget -q --no-proxy ${PADDLE_WHL}
python -m pip install ./\`basename ${PADDLE_WHL}\`

if [[ $REPO == "PaddleSeg" ]]; then
    pip install -e .
    python -m pip install --retries 50 scikit-image
    python -m pip install numba
    python -m pip install sklearn
fi
if [[ $REPO == "PaddleNLP" ]]; then
    python -m pip install --retries 10 paddlenlp
fi
cp \$REPO_PATH/../continuous_integration/tipc/tipc_run.sh .
cp \$REPO_PATH/../continuous_integration/tipc/upload.sh .
cp -r \$REPO_PATH/../continuous_integration/tipc/configs .

bash -x tipc_run.sh
"


# check_status
set +x
echo " "
echo -e "=====================test summary======================"

EXIT_CODE=0

cd $REPO
log_file="RESULT.log"
for f in `find . -name '*.log'`; do
   cat $f | grep "with command" >> $log_file
done

if [[ ! -f ${log_file} ]];then
  echo "[ERROR] ${log_file} not exist, all test cases may fail, please check CI task log"
  EXIT_CODE=8
else
  number_lines=$(cat ${log_file} | wc -l)
  failed_line=$(grep -o "Run failed with command" ${log_file}|wc -l)
  zero=0
  if [ $failed_line -ne $zero ]
  then
      echo "[ERROR] There are $number_lines results in ${log_file}, but failed number of tests is $failed_line."
      echo "The Following Tests Failed: "
      cat ${log_file} | grep "Run failed with command"
      EXIT_CODE=8
  else
      echo "ALL TIPC COMMAND SUCCEED!"
  fi
fi

echo -e "========================================================"
echo " "
echo "Paddle TIPC Tests Finished."
exit ${EXIT_CODE}


# check loss 
log_file="loss.result"
if [[ ! -f ${log_file} ]];then
  echo " "
  echo -e "=====================result summary======================"
  echo "${log_file}: No such file or directory"
  echo "[ERROR] ${log_file} not exist, all test cases may fail, please check CI task log"
  echo "========================================================"
  echo " "
  EXIT_CODE=9
else
  number_lines=$(cat ${log_file} | wc -l)
  failed_line=$(grep "[CHECK]" ${log_file} | grep "False" | wc -l)
  zero=0
  if [ $failed_line -ne $zero ]
  then
      echo " "
      echo "Summary Failed Tests ..."
      echo "[ERROR] There are $number_lines results in ${log_file}, but failed number of tests is $failed_line."
      echo -e "=====================test summary======================"
      echo "The Following Tests Failed: "
      grep "[CHECK]" ${log_file} | grep "False"
      echo -e "========================================================"
      echo " "
      EXIT_CODE=9
  else
      echo "CHECK LOSS SUCCEED!"
  fi
fi



echo "Paddle TIPC Tests Finished."
exit ${EXIT_CODE}
