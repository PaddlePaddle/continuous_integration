#! /bin/bash

echo $CHECK_LOSS
test_mode=${TIPC_MODE:-lite_train_lite_infer}
test_mode=$(echo $test_mode | tr "," "\n")
TIMEOUT=${TIMEOUT:-900} #默认900s(15min)

printmsg()
{
    config_file=$1
    msg="TIMEOUT: ${config_file} time cost > 15min"
    echo $msg
}

run()
{
    config_file=$1
    waitfor=${TIMEOUT}
    command=$*
    $command &
    commandpid=$!
    ( sleep $waitfor ; kill -9 $commandpid >/dev/null 2>&1 && printmsg $config_file ) &
    watchdog=$!
    wait $commandpid >/dev/null 2>&1
    kill -9 $watchdog  >/dev/null 2>&1
}

run_model()
{
    config_file=$1
    mode=$2
    bash test_tipc/prepare.sh $config_file $mode
    bash test_tipc/test_train_inference_python.sh $config_file $mode 
}

echo "grep rules"
if [ ! ${grep_models} ]; then  
    echo "IS NULL"
    grep_models=undefined
    echo ${grep_models}
else  
    echo "NOT NULL"  
    echo ${grep_models}
fi  
if [ ! ${grep_v_models} ]; then  
    echo "IS NULL"  
    grep_v_models=undefined
    echo ${grep_v_models}
else  
    echo "NOT NULL"  
    echo ${grep_v_models}
fi  

find . -name "*train_infer_python.txt" > full_chain_list_all_tmp
if [[ ${grep_models} =~ "undefined" ]]; then
    if [[ ${grep_v_models} =~ "undefined" ]]; then
        cat full_chain_list_all_tmp | sort | uniq > full_chain_list_all
    else
        cat full_chain_list_all_tmp | sort | uniq |grep -v -E ${grep_v_models} > full_chain_list_all  #除了剔除的都跑
    fi
else
    if [[ ${grep_v_models} =~ "undefined" ]]; then
        cat full_chain_list_all_tmp | sort | uniq | grep -E ${grep_models} > full_chain_list_all
    else
        cat full_chain_list_all_tmp | sort | uniq |grep -v -E ${grep_v_models} |grep -E ${grep_models} > full_chain_list_all  #防止选择中含被剔除的模型
    fi
fi

echo "==length models_list=="
wc -l full_chain_list_all #输出本次要跑的模型个数
cat full_chain_list_all #输出本次要跑的模型
cat full_chain_list_all | while read config_file #手动定义
do
# for config_file in `find . -name "*train_infer_python.txt"`; do
start=`date +%s`
    for mode in $test_mode; do
        mode=$(echo $mode | xargs)
        echo "==START=="$config_file"_"$mode
        echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES
        sed -i 's/wget /wget -nv /g' test_tipc/prepare.sh
        run run_model $config_file $mode
        #bash test_tipc/prepare.sh $config_file $mode
        #bash test_tipc/test_train_inference_python.sh $config_file $mode
        bash -x upload.sh ${config_file} ${mode} || echo "upload model error on"`pwd`
        if [[ "$CHECK_LOSS" == "True" ]]; then
            sh check_loss.sh
        fi
        mv test_tipc/output "test_tipc/output_"$(echo $config_file | tr "/" "_")"_"$mode || echo "move output error on "`pwd`
        mv test_tipc/data "test_tipc/data"$(echo $config_file | tr "/" "_")"_"$mode || echo "move data error on "`pwd`
        echo "==END=="$config_file"_"$mode
        sleep 2 #防止显卡未释放
    done
end=`date +%s`
time=`echo $start $end | awk '{print $2-$1-2}'` #减去sleep
echo "${config_file} spend time seconds ${time}"
done

# update model_url latest
if [ -f "tipc_models_url_${REPO}.txt" ];then
    date_stamp=`date +%m_%d`
    push_file=./bce-python-sdk-0.8.27/BosClient.py
    cp "tipc_models_url_${REPO}.txt" "tipc_models_url_${REPO}_latest.txt"
    cp "tipc_models_url_${REPO}.txt" "tipc_models_url_${REPO}_${date_stamp}.txt"
    python2 ${push_file} "tipc_models_url_${REPO}_latest.txt" paddle-qa/fullchain_ce_test/model_download_link
    python2 ${push_file} "tipc_models_url_${REPO}_${date_stamp}.txt" paddle-qa/fullchain_ce_test/model_download_link
fi
