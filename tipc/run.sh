ROOT_PATH=${ROOT_PATH:-/mnt/xly/work}
DOCKER_IMAGE=${DOCKER_IMAGE:-registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82}
DOCKER_IMAGE_PDC=${DOCKER_IMAGE:-registry.baidu.com/paddlecloud/base-images:paddlecloud-ubuntu18.04-gcc8.2-cuda11.0-cudnn8}
#PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Debug_GpuAll_LinuxUbuntu_Gcc82_Cuda10.1_Trton_Py37_Compile_H_DISTRIBUTE_Release/latest/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl}
#PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Debug_GpuAll_LinuxUbuntu_Gcc82_Cuda10.1_Trton_Py37_Compile_H_DISTRIBUTE/latest/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl}
#PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/develop-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-0.0.0.post101-cp37-cp37m-linux_x86_64.whl}
##PADDLE_WHL和FRAME_BRANCH需同步更新
#PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/develop-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-0.0.0.post101-cp37-cp37m-linux_x86_64.whl}
##PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/fullchain_test/whl/paddlepaddle_gpu-0.0.0.post101-cp37-cp37m-linux_x86_64.whl}
#PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/fullchain_test/whl/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl}
#FRAME_BRANCH=${FRAME_BRANCH:-develop}
#PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/release-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-0.0.0.post101-cp37-cp37m-linux_x86_64.whl}
#FRAME_BRANCH=${FRAME_BRANCH:-release/2.3}
PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-LinuxCentos-Gcc82-Cuda102-Trtoff-Py37-Compile/latest/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl}
FRAME_BRANCH=${FRAME_BRANCH:-release/2.4}

work_dir=${ROOT_PATH}/${REPO}
mkdir -p ${work_dir}
cd ${work_dir} && rm -rf *

# download latest tag paddle-wheel
CODE_BOS=https://xly-devops.bj.bcebos.com/PR/Paddle/fullchain_ce_test/${AGILE_PULL_ID}/${AGILE_REVISION}/${REPO}.tar.gz
wget -q --no-proxy ${CODE_BOS} --no-check-certificate
#wget -q --no-proxy https://xly-devops.bj.bcebos.com/PR/Paddle/fullchain_ce_test/${AGILE_PULL_ID}/${AGILE_REVISION}/${REPO}.tar.gz

tar -xpf ${REPO}.tar.gz
cd Paddle
if [[ ${REPO} == "models" ]]
then
cp -r models/tutorials/mobilenetv3_prod/Step6 ./
rm -rf models
mv Step6 models
fi

#add upload
wget -q --no-proxy -O $PWD/bce_whl.tar.gz  https://paddle-docker-tar.bj.bcebos.com/home/bce_whl.tar.gz --no-check-certificate

#python -m pip install paddleseg
#cp continuous_integration/tipc/tipc.sh .
#cp continuous_integration/tipc/checkout_result.sh .
cp -r continuous_integration/tipc/* .
cd ${REPO}
if [[ $REPO == "PaddleNLP" ]]; then
    cd tests
fi
cp ${ROOT_PATH}/db_info.yaml ./
cp ${ROOT_PATH}/icafe_conf.py ./
cp ${work_dir}/Paddle/continuous_integration/tipc/checkout_result.sh ./
cp ${work_dir}/Paddle/continuous_integration/tipc/report.py ./
cd ${work_dir}/Paddle
if [[ ${CHAIN} == "chain_distribution" ]]
then
    cd ${REPO}
    if [[ $REPO == "PaddleNLP" ]]; then
        cd tests
    fi
    cp -r ../continuous_integration/tipc/* .
    cp ${ROOT_PATH}/config.ini .
    cp ${ROOT_PATH}/pdc.sh .
    cp ${ROOT_PATH}/pdc_conf.ini .
    cd ${work_dir}/Paddle
    bash tipc.sh ${REPO} ${CHAIN} ${PADDLE_WHL} ${DOCKER_IMAGE} ${CODE_BOS} ${FRAME_BRANCH} ${SENDER} ${RECVIER} ${MAIL_PROXY}  
else
    bash tipc.sh ${REPO} ${CHAIN} ${PADDLE_WHL} ${DOCKER_IMAGE} ${CODE_BOS} ${FRAME_BRANCH} ${SENDER} ${RECVIER} ${MAIL_PROXY}
fi

cd ${REPO}
if [[ $REPO == "PaddleNLP" ]]; then
    cd tests
fi
bash checkout_result.sh
