#! /bin/bash

echo $CHECK_LOSS
test_mode=${TIPC_MODE:-lite_train_lite_infer}
test_mode=$(echo $test_mode | tr "," "\n")

for config_file in `find . -name "*train_infer_python.txt"`; do
    for mode in $test_mode; do
        mode=$(echo $mode | xargs)
        echo "==START=="$config_file"_"$mode
        echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES
        bash test_tipc/prepare.sh $config_file $mode
        bash test_tipc/test_train_inference_python.sh $config_file $mode
        mv test_tipc/output "test_tipc/output_"$(echo $config_file | tr "/" "_")"_"$mode || echo "move output error on "`pwd`
        mv test_tipc/data "test_tipc/data"$(echo $config_file | tr "/" "_")"_"$mode || echo "move data error on "`pwd`
        bash -x upload.sh ${config_file} ${mode} || echo "upload model error on"`pwd`
        if [[ "$CHECK_LOSS" == "True" ]]; then
            sh check_loss.sh
        fi
        echo "==END=="$config_file"_"$mode
    done
done
