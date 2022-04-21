#REPO=$1
#AGILE_PULL_ID=$2
#AGILE_REVISION=$3
ROOT_PATH=${ROOT_PATH:-/mnt/xly/work}

work_dir=${ROOT_PATH}/${REPO}
mkdir -p ${work_dir}
cd ${work_dir} && rm -rf *

# download latest tag paddle-wheel
wget -q --no-proxy https://xly-devops.bj.bcebos.com/PR/Paddle/fullchain_ce_test/${AGILE_PULL_ID}/${AGILE_REVISION}/${REPO}.tar.gz >/dev/null

tar -xpf ${REPO}.tar.gz
cd Paddle

#add upload
wget -q --no-proxy -O $PWD/bce_whl.tar.gz  https://paddle-docker-tar.bj.bcebos.com/home/bce_whl.tar.gz --no-check-certificate  >/dev/null

#python -m pip install paddleseg
cp continuous_integration/tipc/tipc.sh .
#export TIPC_MODE="whole_train_whole_infer"
export CHECK_LOSS=True
sh tipc.sh ${REPO}
