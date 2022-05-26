#!/bin/bash
set -xe

server="paddlecloud.baidu-int.com"
port=80

#cloud paramters
num_trainers=2
num_cards=2

card_type=${card_type:-"gpu_v100"}
num_trainers=${num_trainers:-1}
num_cards=${num_cards:-2}
all_cards=4

JOB_MANE=$1
REPO=$2
PADDLE_WHL=$3
DOCKER_IMAGE=$4
CODE_BOS=$5
CONFIG_FILE=$6
MODE=$7

k8s_wall_time="00:00:00"

distribute=" --k8s-not-local --distribute-job-type NCCL2 "

paddlecloud job \
    --server ${server} \
    --port ${port} \
    train --job-version paddle-fluid-custom \
    --group-name dltp-32g-0-yq01-k8s-gpu-v100-8-0      \
    --cluster-name v100-32-0-cluster \
    --k8s-gpu-cards ${num_cards} \
    --k8s-priority high \
    --k8s-wall-time ${k8s_wall_time} \
    --k8s-memory 350Gi \
    --job-name ${JOB_MANE} \
    --permission group \
    --start-cmd "bash distribution.sh ${REPO} ${CODE_BOS} ${PADDLE_WHL} ${CONFIG_FILE} ${MODE}" \
    --files distribution.sh \
    --job-conf config.ini \
    --k8s-trainers ${num_trainers} ${distribute} \
    --is-auto-over-sell 0 \
    --image-addr "${DOCKER_IMAGE}" \
    --k8s-cpu-cores 35


