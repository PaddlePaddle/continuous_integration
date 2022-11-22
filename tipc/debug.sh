MODEL_NAME=$1
MODEL_COMMIT=$2
FRAME_COMMIT=$3
REPO=$4
CHAIN=$5
FRAME_BRANCH=$6
DOCKER_IMAGE=$7
CODE_BOS=$8
SENDER=$9
RECVIER=${10}
MAIL_PROXY=${11}
FRAME_PATH=${12}
PADDLE_WHL=https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-LinuxCentos-Gcc82-Cuda102-Trtoff-Py37-Compile/${FRAME_COMMIT}/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
if [[ ${FRAME_BRANCH} =~ "develop"]]; then
  PADDLE_WHL=https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-LinuxCentos-Gcc82-Cuda102-Trtoff-Py37-Compile/${FRAME_COMMIT}/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
fi
echo "---------------------------"
echo "bash -x tipc_run.sh ${REPO} ${CHAIN} ${PADDLE_WHL} ${FRAME_BRANCH} ${DOCKER_IMAGE} ${CODE_BOS} ${SENDER} ${RECVIER} ${MAIL_PROXY}"

export grep_models=${MODEL_NAME}
export PADDLE_WHL=${PADDLE_WHL}
export DEBUG=True

if [[ ! -d ${REPO}_base ]]
then
  git clone https://github.com/PaddlePaddle/${REPO} ${REPO}_base
  cd ${REPO}_base
  if [[ ${REPO} == PaddleOCR ]]
  then
    git checkout dygraph
  elif [[ ${REPO} == PaddleRec ]]
  then
    git checkout master
  else
    git checkout develop
  fi
  cd -
fi
rm -rf ${REPO}
cp -r ${REPO}_base ${REPO}
cd ${REPO}
git checkout ${MODEL_COMMIT}
cp -rf ${FRAME_PATH}/continuous_integration/tipc/* ./
sed -i 's#gpu_list:0|0,1#gpu_list:2|2,3#g' test_tipc/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco_KL_train_infer_python.txt
#cd -

pip uninstall paddlepaddle-gpu -y
pip install -U ${PADDLE_WHL}

bash -x tipc_run.sh ${REPO} ${CHAIN} ${PADDLE_WHL} ${FRAME_BRANCH} ${DOCKER_IMAGE} ${CODE_BOS} ${SENDER} ${RECVIER} ${MAIL_PROXY}

EXIT_CODE=0
log_file=RESULT
if [[ ! -f ${log_file} ]];then
  EXIT_CODE=2
elif [[ ! -s ${log_file} ]];then
  EXIT_CODE=3
else
  number_lines=$(cat ${log_file} | wc -l)
  failed_line=$(grep -o "Run failed with command" ${log_file}|wc -l)
  if [[ $failed_line -ne $zero ]]
  then
    EXIT_CODE=1
  fi
fi

cd ..
mv ${REPO} ${REPO}_${FRAME_COMMIT}_${MODEL_COMMIT}

exit ${EXIT_CODE}
