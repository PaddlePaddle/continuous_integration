#!/usr/bin/env bash

set -eo pipefail

ROOT=`dirname "$0"`
ROOT=`cd "$ROOT/.."; pwd`
export OUTPUT=$ROOT/output
export OUTPUT_BIN=$ROOT/build
export DATA_ROOT=$ROOT/data
export TOOLS_ROOT=$ROOT/tools
export CASE_ROOT=$ROOT/bin
export gpu_type=$1

mkdir -p $DATA_ROOT
cd $DATA_ROOT
if [ ! -f PaddleClas/infer_static/AlexNet/__model__ ]; then
    echo "==== Download PaddleClas data and models ===="
    wget --no-proxy -q https://sys-p0.bj.bcebos.com/Paddle-UnitTest-Model/PaddleClas.tgz --no-check-certificate
    tar -zxf PaddleClas.tgz
fi

if [ ! -f PaddleDetection/infer_static/yolov3_darknet/__model__ ]; then
    echo "==== Download PaddleDetection data and models ===="
    wget --no-proxy -q https://sys-p0.bj.bcebos.com/Paddle-UnitTest-Model/PaddleDetection.tgz --no-check-certificate
    tar -zxf PaddleDetection.tgz
fi

#if [ ! -f PaddleOCR/ch_ppocr_mobile_v1.1_cls_infer/model ]; then
#    echo "==== Download PaddleOCR data and models ===="
#    wget --no-proxy -q https://sys-p0.bj.bcebos.com/Paddle-UnitTest-Model/PaddleOCR.tgz --no-check-certificate
#    tar -zxf PaddleOCR.tgz
#fi
#
#if [ ! -f PaddleSeg/infer_static/deeplabv3p/__model__ ]; then
#    echo "==== Download PaddleSeg data and models ===="
#    wget --no-proxy -q https://sys-p0.bj.bcebos.com/Paddle-UnitTest-Model/PaddleSeg.tgz --no-check-certificate
#    tar -zxf PaddleSeg.tgz
#fi

cd -

# bash $CASE_ROOT/pd-yolo-test-jetson.sh #迁移至paddletest仓库
bash $CASE_ROOT/pd-clas-test-jetson-trt.sh
bash $CASE_ROOT/pd-rcnn-test-jetson-trt.sh

