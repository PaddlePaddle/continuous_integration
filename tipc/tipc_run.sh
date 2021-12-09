#! /bin/bash


test_mode=${TIPC_MODE:-lite_train_lite_infer}
test_mode=$(echo $test_mode | tr "," "\n")

for config_file in `find . -name "*train_infer_python.txt"`; do
    for mode in $test_mode; do
        mode=$(echo $mode | xargs)
        echo "======="$config_file"==========="
        echo "======="$mode"==========="
        echo "CUDA_VISIBLE_DEVICES="$CCUDA_VISIBLE_DEVICES"======="
        bash test_tipc/prepare.sh $config_file $mode
        bash test_tipc/test_train_inference_python.sh $config_file $mode
    done
done
