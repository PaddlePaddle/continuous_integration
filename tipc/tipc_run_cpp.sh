#! /bin/bash


test_mode=${TIPC_MODE:-lite_train_lite_infer}
test_mode=$(echo $test_mode | tr "," "\n")


cd deploy/cpp_infer
mkdir -p Paddle/build
wget --no-proxy https://paddle-inference-lib.bj.bcebos.com/2.2.2/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz
tar -zxf paddle_inference.tgz
mv paddle_inference Paddle/build/paddle_inference_install_dir
cd -


for config_file in `find . -name "*_infer_cpp_*.txt"`; do
    for mode in $test_mode; do
        mode=$(echo $mode | xargs)
        echo "==START=="$config_file"_"$mode
        echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES
        export http_proxy=http://172.19.56.199:3128
        export https_proxy=http://172.19.56.199:3128
        export no_proxy=bcebos.com
        bash -ex test_tipc/prepare.sh $config_file "cpp_infer"
        bash -ex test_tipc/test_inference_cpp.sh $config_file
        echo "==END=="$config_file"_"$mode
    done
done
