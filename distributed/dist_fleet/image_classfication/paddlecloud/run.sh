#!/bin/bash
# 线上环境
server="paddlecloud.baidu-int.com"
port=80

user_ak="3cf90cc276085c14a4c908b975942edb"
user_sk="b50a7d1cd8ad5bcc8cf18388f8de8bb6"

# 作业参数
job_name=ResNet50_benchmark
card_type=gpu_v100
num_trainers=4
num_cards=8
k8s_wall_time="96:00:00"
image_addr="registry.baidu.com/paddlepaddle-public/distributed_paddle_dgc:cuda9_cudnn7"
distribute=" --k8s-not-local --distribute-job-type NCCL2 "
if [[ $num_trainers == "1" ]]; then
    distribute=""
    enable_dgc=False
fi

paddlecloud job \
    --server ${server} \
    --port ${port} \
    --user-ak ${user_ak} \
    --user-sk ${user_sk} \
    train --job-version paddle-fluid-custom \
    --k8s-gpu-type baidu/${card_type} \
    --cluster-name paddle-jpaas-ai00-gpu \
    --k8s-gpu-cards ${num_cards} \
    --k8s-priority high \
    --k8s-wall-time ${k8s_wall_time} \
    --k8s-memory 190Gi \
    --job-name ${job_name} \
    --start-cmd "bash train_pretrain.sh" \
    --job-conf config.ini \
    --files  end_hook.sh before_hook.sh ../*.py ../models/*.py ../utils/*.py train_pretrain.sh \
    --k8s-trainers ${num_trainers} ${distribute} \
    --k8s-cpu-cores 35 \
    --image-addr "${image_addr}"
