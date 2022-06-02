#! /bin/bash

function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}

function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}

function printmsg()
{
    model_name=$1
    config_file=$2
    msg="${model_name} ${config_file} time cost > ${time_out}seconds"
    echo $msg >> TIMEOUT
}

function run()
{
    waitfor=${time_out}
    command=$*
    $command &
    commandpid=$!
    ( sleep $waitfor ; kill -9 $commandpid >/dev/null 2>&1 && printmsg $2 ) &
    watchdog=$!
    wait $commandpid >/dev/null 2>&1
    kill -9 $watchdog  >/dev/null 2>&1
}

#if [[ ${REPO} == "PaddleOCR" ]]
#then
#sed -i '192 i if [ ! -d "paddle_inference" ]; then' test_tipc/test_inference_cpp.sh
#sed -i '193 i ln -s paddle_inference_install_dir paddle_inference' test_tipc/test_inference_cpp.sh
#sed -i '194 i fi' test_tipc/test_inference_cpp.sh
#fi

function run_model()
{
    config_file=$1
    mode=$2
    case $CHAIN in
    chain_base)
        bash test_tipc/prepare.sh $config_file $mode
        bash test_tipc/test_train_inference_python.sh $config_file $mode 
        ;;
    chain_infer_cpp)
        bash test_tipc/prepare.sh $config_file $mode $PADDLE_INFERENCE_TGZ
        bash test_tipc/test_inference_cpp.sh $config_file 
        ;;
    chain_amp)
        bash test_tipc/prepare.sh $config_file $mode
        bash test_tipc/test_train_inference_python.sh $config_file $mode 
        ;;
    chain_serving_cpp)
        #build server
        rm -f ${Serving_repo_path}/core/general-server/op/general_clas_op.*
        rm -f ${Serving_repo_path}/core/predictor/tools/pp_shitu_tools/preprocess_op.*
        cp deploy/serving_cpp/preprocess/general_clas_op.* ${Serving_repo_path}/core/general-server/op
        cp deploy/serving_cpp/preprocess/preprocess_op.* ${Serving_repo_path}/core/predictor/tools/pp_shitu_tools

        cd Serving/
        rm -rf server-build-gpu-opencv
        mkdir server-build-gpu-opencv && cd server-build-gpu-opencv
        set http_proxy=${HTTP_PROXY}
        set https_proxy=${HTTPS_PROXY}
        cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
            -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
            -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
            -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH} \
            -DCUDNN_LIBRARY=${CUDNN_LIBRARY} \
            -DCUDA_CUDART_LIBRARY=${CUDA_CUDART_LIBRARY} \
            -DTENSORRT_ROOT=${TENSORRT_LIBRARY_PATH} \
            -DOPENCV_DIR=${OPENCV_DIR} \
            -DWITH_OPENCV=ON \
            -DSERVER=ON \
            -DWITH_GPU=ON ..
        make -j32

        python -m pip install python/dist/paddle*
        export SERVING_BIN=$PWD/core/general-server/serving
        unset http_proxy
        unset https_proxy
        cd  ../../

        #run models
        bash test_tipc/prepare.sh $config_file $mode
        bash test_tipc/test_serving_infer_cpp.sh $config_file $mode 
        ;;
    chain_serving_python)
        #pip install paddle-serving-server-gpu==0.8.3.post101
        #pip install paddle_serving_client==0.8.3
        #pip install paddle-serving-app==0.8.3
        bash test_tipc/prepare.sh $config_file $mode
        bash test_tipc/test_serving_infer_python.sh $config_file $mode
        ;;
    chain_paddle2onnx)
        bash test_tipc/prepare.sh $config_file $mode
        bash test_tipc/test_paddle2onnx.sh $config_file $mode 
        ;;
    chain_ptq_infer_python)
        bash test_tipc/prepare.sh $config_file $mode
        bash test_tipc/test_train_inference_python.sh $config_file $mode
        ;;
    chain_pact_infer_python)
        bash test_tipc/prepare.sh $config_file $mode
        bash test_tipc/test_train_inference_python.sh $config_file $mode
        ;;
    *)
        echo "CHAIN must be chain_base chain_infer_cpp chain_amp chain_serving_cpp chain_serving_python chain_paddle2onnx chain_distribution chain_pact_infer_python chain_ptq_infer_python"
        echo "$CHAIN not supported at the moment"
        exit 2
        ;;
 
    esac
}

if [[ $REPO == "PaddleOCR" ]]; then
sed -i "s/GPUID=\$2/GPUID=\$3/g" test_tipc/test_serving_infer_python.sh
sed -i "s/web_service|pipeline/web_service/g" test_tipc/test_serving_infer_python.sh
fi

mkdir -p test_tipc/output
touch TIMEOUT
if [[ $CHAIN == "chain_paddle2onnx" ]]; then
    pip install onnx==1.9.0
fi

if [[ $CHAIN == "chain_serving_python" ]]; then
    pip install paddle-serving-server-gpu==0.8.3.post101
    pip install paddle_serving_client==0.8.3
    pip install paddle-serving-app==0.8.3
fi

if [[ $CHAIN == chain_serving_cpp ]]; then
        # 安装client 和 app
        pip install paddle_serving_client
        pip install paddle-serving-app

        # 准备server的编译环境
        apt-get update
        apt install -y libcurl4-openssl-dev libbz2-dev
        wget -nv https://paddle-serving.bj.bcebos.com/others/centos_ssl.tar && tar xf centos_ssl.tar && rm -rf centos_ssl.tar && mv libcrypto.so.1.0.2k /usr/lib/libcrypto.so.1.0.2k && mv libssl.so.1.0.2k /usr/lib/libssl.so.1.0.2k && ln -sf /usr/lib/libcrypto.so.1.0.2k /usr/lib/libcrypto.so.10 && ln -sf /usr/lib/libssl.so.1.0.2k /usr/lib/libssl.so.10 && ln -sf /usr/lib/libcrypto.so.10 /usr/lib/libcrypto.so && ln -sf /usr/lib/libssl.so.10 /usr/lib/libssl.so
        rm -rf /usr/local/go
        wget -nv -qO- https://paddle-ci.cdn.bcebos.com/go1.17.2.linux-amd64.tar.gz | tar -xz -C /usr/local
        export GOROOT=/usr/local/go
        export GOPATH=/root/gopath
        export PATH=$PATH:$GOPATH/bin:$GOROOT/bin
        go env -w GO111MODULE=on
        go env -w GOPROXY=https://goproxy.cn,direct
        go install github.com/grpc-ecosystem/grpc-gateway/protoc-gen-grpc-gateway@v1.15.2
        go install github.com/grpc-ecosystem/grpc-gateway/protoc-gen-swagger@v1.15.2
        go install github.com/golang/protobuf/protoc-gen-go@v1.4.3
        go install google.golang.org/grpc@v1.33.0
        go env -w GO111MODULE=auto

        wget -nv https://paddle-qa.bj.bcebos.com/PaddleServing/opencv3.tar.gz && tar -xvf opencv3.tar.gz && rm -rf opencv3.tar.gz
        export OPENCV_DIR=$PWD/opencv3

        # clone Serving
        set http_proxy=${HTTP_PROXY}
        set https_proxy=${HTTPS_PROXY}
        git clone https://github.com/PaddlePaddle/Serving.git -b v0.8.3 --depth=1
        cd Serving
        export Serving_repo_path=$PWD
        git submodule update --init --recursive
        set http_proxy=
        set https_proxy=

        python -m pip install -r python/requirements.txt

        export PYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
        export PYTHON_LIBRARIES=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
        export PYTHON_EXECUTABLE=`which python`

        export CUDA_PATH='/usr/local/cuda'
        export CUDNN_LIBRARY='/usr/local/cuda/lib64/'
        export CUDA_CUDART_LIBRARY='/usr/local/cuda/lib64/'
        export TENSORRT_LIBRARY_PATH='/usr/local/TensorRT6-cuda10.1-cudnn7/targets/x86_64-linux-gnu/'
        cd ..
fi

# 确定链条的txt、mode、timeout
case $CHAIN in
chain_base) 
    file_txt=train_infer_python.txt
    mode=lite_train_lite_infer
    time_out=600
    ;;
chain_infer_cpp)
    file_txt=*_infer_cpp_*
    if [[ ${REPO} == "PaddleSeg" ]]
    then
        file_txt=*inference_cpp.txt*
    fi
    mode=cpp_infer
    time_out=600
    ;;
chain_amp)
    file_txt=*_amp_*
    mode=lite_train_lite_infer
    time_out=600
    ;;
chain_serving_cpp)
    file_txt=*_serving_cpp_*
    mode=serving_infer
    time_out=600
    ;;
chain_serving_python)
    file_txt=*_serving_python_*
    mode=serving_infer
    time_out=60
    ;;
chain_paddle2onnx)
    file_txt=*paddle2onnx*
    mode=paddle2onnx_infer
    time_out=60
    ;;
chain_distribution)
    file_txt=*_fleet_*
    mode=lite_train_lite_infer
    time_out=600
    ;;
chain_pact_infer_python)
    file_txt=train_pact_infer_python.txt
    mode=lite_train_lite_infer
    time_out=600
    ;;
chain_ptq_infer_python)
    file_txt=train_ptq_infer_python.txt
    mode=whole_infer
    time_out=60
    ;;
*)
    echo "CHAIN must be chain_base chain_infer_cpp chain_amp chain_serving_cpp chain_serving_python chain_paddle2onnx chain_distribution chain_pact_infer_python chain_ptq_infer_python"
    echo "$CHAIN not supported at the moment"
    exit 1
    ;;
esac

# 确定套件的待测模型列表, 其txt保存到full_chain_list_all
python model_list.py $REPO ${PWD}/test_tipc/configs/ $file_txt full_chain_list_all_tmp 
if [ ! ${grep_models} ]; then  
    grep_models=undefined
fi  
if [ ! ${grep_v_models} ]; then  
    grep_v_models=undefined
fi  
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
        cat full_chain_list_all_tmp | sort | uniq |grep -v -E ${grep_v_models} |grep -E ${grep_models} > full_chain_list_all
    fi
fi
echo "==length models_list=="
wc -l full_chain_list_all #输出本次要跑的模型个数
cat full_chain_list_all #输出本次要跑的模型

# 跑模型
sed -i 's/wget /wget -nv /g' test_tipc/prepare.sh
cat full_chain_list_all | while read config_file
do
  dataline=$(awk 'NR==1, NR==32{print}'  $config_file)
  IFS=$'\n'
  lines=(${dataline})
  model_name=$(func_parser_value "${lines[1]}")
  if [[ $CHAIN == "chain_distribution" ]]
  then
    echo "==START=="$config_file
    JOB_NAME=tipc-${model_name}-${mode}
    PADDLE_WHL=$3
    DOCKER_IMAGE=$4
    CODE_BOS=$5
    sh pdc.sh ${JOB_NAME} ${REPO} ${PADDLE_WHL} ${DOCKER_IMAGE} ${CODE_BOS} $config_file ${mode} ${time_out} >log.pdc 2>&1
    pdc_job_id=`cat log.pdc | grep "jobId = job-" | awk -F ',' '{print $1}' | awk -F '= ' '{print $2}'`
    # todo 判断pdc任务是否提交成功
    echo ${model_name},${pdc_job_id} >> pdc_job_id
  else
    start=`date +%s`
    echo "==START=="$config_file
    run run_model $config_file $mode $time_out $model_name

    echo "==END=="$config_file
    sleep 2 #防止显卡未释放
    end=`date +%s`
    time=`echo $start $end | awk '{print $2-$1-2}'` #减去sleep
    echo "${config_file} spend time seconds ${time}"

    if [[ "${DEBUG}" == "True" ]]
    then
      bash -x upload.sh ${config_file} ${mode} ${CHAIN} || echo "upload model error on"`pwd`
    fi
    mv test_tipc/output "test_tipc/output_"$(echo $config_file | tr "/" "_")"_"$mode || echo "move output error on "`pwd`
    mv test_tipc/data "test_tipc/data"$(echo $config_file | tr "/" "_")"_"$mode || echo "move data error on "`pwd`
  fi
done

# watch_job_status and get log, job_id in file pdc_job_id
if [[ "$CHAIN" == "chain_distribution" ]]
then
  export http_proxy=
  export https_proxy=
  python get_pdc_job_result.py pdc_job_id
fi


# update model_url latest
if [ -f "tipc_models_url_${REPO}.txt" ];then
    date_stamp=`date +%m_%d`
    push_file=./bce-python-sdk-0.8.27/BosClient.py
    cp "tipc_models_url_${REPO}_${CHAIN}.txt" "tipc_models_url_${REPO}_${CHAIN}_latest.txt"
    cp "tipc_models_url_${REPO}_${CHAIN}.txt" "tipc_models_url_${REPO}_${CHAIN}_${date_stamp}.txt"
    python2 ${push_file} "tipc_models_url_${REPO}_${CHAIN}_latest.txt" paddle-qa/fullchain_ce_test/model_download_link
    python2 ${push_file} "tipc_models_url_${REPO}_${CHAIN}_${date_stamp}.txt" paddle-qa/fullchain_ce_test/model_download_link
fi
