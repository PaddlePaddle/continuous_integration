#!/usr/bin/env bash
#run_tools9.sh {model_name} ${cards}
#export repo_path=/paddle/deploy/tools/tool-9/
export repo_path=$PWD
export tools_path=$PWD
export TRAIN_LOG_DIR=${tools_path}/tool_log
rm -rf ${TRAIN_LOG_DIR}
mkdir -p ${TRAIN_LOG_DIR}

unset GREP_OPTIONS
cards=${2:-"1"}
if [ $cards = "1" ];then
     export CUDA_VISIBLE_DEVICES=0
elif [ $cards = "8" ];then
     export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
     echo "cards error";
     exit 1;
fi
# 运行模型
case $1 in
    ResNet50_vd)  model_sh=run_ResNet50_vd.sh
                  model_path=${repo_path}/PaddleClas/ ;;
    MobileNetV1)  model_sh=run_MobileNetV1.sh
                  model_path=${repo_path}/PaddleClas/ ;;
    yolov3)       model_sh=run_yolov3.sh
                  model_path=${repo_path}/PaddleDetection/ ;;
    *) echo "model name error"; exit 1;
esac
cd ${model_path}/
rm -rf ${model_sh}
cp ${tools_path}/${model_sh} ${model_path}/
# 统计GPU显存占用
rm -rf  gpu_use.log
gpu_id=`echo $CUDA_VISIBLE_DEVICES | cut -c1`
nvidia-smi --id=$gpu_id --query-gpu=utilization.gpu,memory.used --format=csv -lms 100 > gpu_use.log 2>&1 &
gpu_memory_pid=$!
# 统计CPU
time=$(date "+%Y-%m-%d %H:%M:%S")
LAST_CPU_INFO=$(cat /proc/stat | grep -w cpu | awk '{print $2,$3,$4,$5,$6,$7,$8}')
LAST_SYS_IDLE=$(echo $LAST_CPU_INFO | awk '{print $4}')
LAST_TOTAL_CPU_T=$(echo $LAST_CPU_INFO | awk '{print $1+$2+$3+$4+$5+$6+$7}')

bash ${model_sh} ${cards}
#sleep ${TIME_INTERVAL}   # 执行模型的时间
NEXT_CPU_INFO=$(cat /proc/stat | grep -w cpu | awk '{print $2,$3,$4,$5,$6,$7,$8}')
NEXT_SYS_IDLE=$(echo $NEXT_CPU_INFO | awk '{print $4}')
NEXT_TOTAL_CPU_T=$(echo $NEXT_CPU_INFO | awk '{print $1+$2+$3+$4+$5+$6+$7}')

#系统空闲时间
SYSTEM_IDLE=`echo ${NEXT_SYS_IDLE} ${LAST_SYS_IDLE} | awk '{print $1-$2}'`
#CPU总时间
TOTAL_TIME=`echo ${NEXT_TOTAL_CPU_T} ${LAST_TOTAL_CPU_T} | awk '{print $1-$2}'`
#echo "LAST_SYS_IDLE:" $LAST_SYS_IDLE
#echo "NEXT_SYS_IDLE:" $NEXT_SYS_IDLE
#echo "LAST_TOTAL_CPU_T:" $LAST_TOTAL_CPU_T
#echo "NEXT_TOTAL_CPU_T:" $NEXT_TOTAL_CPU_T
#echo "SYSTEM_IDLE:" $SYSTEM_IDLE
#echo "TOTAL_TIME: " $TOTAL_TIME
if [ $TOTAL_TIME == 0 ];then  # 两次系统的总时间一致,说明CPU的使用的时间计划为0
    AVG_CPU_USE=0
else
    CPU_USAGE=`echo ${SYSTEM_IDLE} ${TOTAL_TIME} | awk '{printf "%.2f", (1-$1/$2)*100}'`
    AVG_CPU_USE=${CPU_USAGE}
fi

# 计算显存占用
kill ${gpu_memory_pid}
MAX_GPU_MEMORY_USE=`awk 'BEGIN {max = 0} {if(NR>1){if ($3 > max) max=$3}} END {print max}' gpu_use.log`
AVG_GPU_USE=`awk '{if(NR>1 && $1 >0){time+=$1;count+=1}} END{if(count>0) avg=time/count; else avg=0; printf("%.2f\n" ,avg)}' gpu_use.log`

echo "{AVG_CPU_USE: $AVG_CPU_USE %,MAX_GPU_MEMORY_USE: $MAX_GPU_MEMORY_USE MiB, AVG_GPU_USE:$AVG_GPU_USE %}"
