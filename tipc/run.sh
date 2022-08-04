ROOT_PATH=${ROOT_PATH:-/mnt/xly/work}
DOCKER_IMAGE=${DOCKER_IMAGE:-registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82}
DOCKER_IMAGE_PDC=${DOCKER_IMAGE:-registry.baidu.com/paddlecloud/base-images:paddlecloud-ubuntu18.04-gcc8.2-cuda11.0-cudnn8}
#PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Debug_GpuAll_LinuxUbuntu_Gcc82_Cuda10.1_Trton_Py37_Compile_H_DISTRIBUTE_Release/latest/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl}
#PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Debug_GpuAll_LinuxUbuntu_Gcc82_Cuda10.1_Trton_Py37_Compile_H_DISTRIBUTE/latest/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl}
#PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/develop-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-0.0.0.post101-cp37-cp37m-linux_x86_64.whl}
##PADDLE_WHL和FRAME_BRANCH需同步更新
#PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/develop-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-0.0.0.post101-cp37-cp37m-linux_x86_64.whl}
#FRAME_BRANCH=${FRAME_BRANCH:-develop}
PADDLE_WHL=${PADDLE_WHL:-https://paddle-qa.bj.bcebos.com/release-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-0.0.0.post101-cp37-cp37m-linux_x86_64.whl}
FRAME_BRANCH=${FRAME_BRANCH:-release/2.3}

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
if [[ ${CHAIN} == "chain_distribution" ]]
then
    cd ${REPO}
    cp -r ../continuous_integration/tipc/* .
    cp ${ROOT_PATH}/config.ini .
    cp ${ROOT_PATH}/pdc.sh .
    sh tipc_run.sh ${REPO} ${CHAIN} ${PADDLE_WHL} ${DOCKER_IMAGE_PDC} ${CODE_BOS} 
    cd ..
else
    sh tipc.sh ${REPO} ${CHAIN} ${PADDLE_WHL} ${DOCKER_IMAGE}
fi

cd $REPO
log_file="RESULT"
for f in `find . -name '*.log'`; do
   cat $f | grep "with command" >> $log_file
done
cp ../continuous_integration/tipc/checkout_result.sh ./
cp ../continuous_integration/tipc/report.py ./
python report.py ${REPO} ${CHAIN} ${SENDER} ${RECVIER} ${MAIL_PROXY}
sh checkout_result.sh

#-------------------------------------------------
### baobiao
#cd $REPO
repo=$REPO
repo_branch=`git branch | awk '{print $2}'`
repo_commit=`git log | head -1 | awk '{print $2}'`
chain=$CHAIN
paddle_whl=$PADDLE_WHL
docker_image=$DOCKER_IMAGE
if [[ $CHAIN == "chain_distribution" ]]
then
    docker_image=$DOCKER_IMAGE_PDC
fi
#frame_branch: 
#frame_commit: 需安装确定, 在tipc.sh和distribution.sh中安装后判断，并将结果保存>到本地文件，这里读文件获取
#cuda: 同frame_commit
#cudnn： 同frame_commit
frame_branch=$FRAME_BRANCH
frame_commit=`head -1 paddle_info`
cuda=`head -2 paddle_info | tail -1`
cudnn=`tail -1 paddle_info`
python=3.7 #只支持3.7

echo $repo
echo $repo_branch
echo $repo_commit
echo $CHAIN
echo $paddle_whl
echo $docker_image
echo $frame_branch
echo $frame_commit
echo $cuda
echo $cudnn
echo $python

#model 需遍历结果来确定 
#stag 需遍历结果来确定
#cmd 需遍历结果来确定
#status 需遍历结果来确定
#icafe 需遍历结果来确定,自动创建icafe卡片，返回icafe地址
#log 需遍历结果来确定

# 结合report.py入库
