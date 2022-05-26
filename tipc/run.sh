ROOT_PATH=${ROOT_PATH:-/mnt/xly/work}
#DOCKER_IMAGE=${DOCKER_IMAGE:-registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82}
DOCKER_IMAGE=${DOCKER_IMAGE:-registry.baidu.com/paddlecloud/base-images:paddlecloud-ubuntu18.04-gcc8.2-cuda11.0-cudnn8}
PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Debug_GpuAll_LinuxUbuntu_Gcc82_Cuda10.1_Trton_Py37_Compile_H_DISTRIBUTE_Release/latest/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl}

work_dir=${ROOT_PATH}/${REPO}
mkdir -p ${work_dir}
cd ${work_dir} && rm -rf *

# download latest tag paddle-wheel
CODE_BOS=https://xly-devops.bj.bcebos.com/PR/Paddle/fullchain_ce_test/${AGILE_PULL_ID}/${AGILE_REVISION}/${REPO}.tar.gz
wget -q --no-proxy ${CODE_BOS} --no-check-certificate
#wget -q --no-proxy https://xly-devops.bj.bcebos.com/PR/Paddle/fullchain_ce_test/${AGILE_PULL_ID}/${AGILE_REVISION}/${REPO}.tar.gz

tar -xpf ${REPO}.tar.gz
cd Paddle
cp -r models/tutorials/mobilenetv3_prod/Step6 ./
rm -rf models
mv Step6 models

#add upload
wget -q --no-proxy -O $PWD/bce_whl.tar.gz  https://paddle-docker-tar.bj.bcebos.com/home/bce_whl.tar.gz --no-check-certificate

#python -m pip install paddleseg
#cp continuous_integration/tipc/tipc.sh .
if [[ ${CHAIN} == "chain_distribution" ]]
then
    cd ${REPO}
    cp -r ../continuous_integration/tipc/* .
    sh tipc_run.sh ${REPO} ${CHAIN} ${DOCKER_IMAGE} ${PADDLE_WHL} ${CODE_BOS} 
else
    sh tipc.sh ${REPO} ${CHAIN}
fi
