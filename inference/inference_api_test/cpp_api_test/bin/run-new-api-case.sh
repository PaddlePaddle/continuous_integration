#!/usr/bin/env bash

set -eo pipefail

ROOT=`dirname "$0"`
ROOT=`cd "$ROOT/.."; pwd`
export OUTPUT=$ROOT/output
export OUTPUT_BIN=$ROOT/build
export DATA_ROOT=$ROOT/data
export TOOLS_ROOT=$ROOT/tools
export CASE_ROOT=$ROOT/bin

mkdir -p $DATA_ROOT
cd $DATA_ROOT
if [ ! -f PaddleClas/infer_static/AlexNet/__model__ ]; then
    echo "==== Download data and models ===="
    wget --no-proxy -q https://sys-p0.bj.bcebos.com/Paddle-UnitTest-Model/PaddleClas.tgz --no-check-certificate
    tar -zxf PaddleClas.tgz
fi

if [ ! -f PaddleDetection/infer_static/yolov3_darknet/__model__ ]; then
    echo "==== Download data and models ===="
    wget --no-proxy -q https://sys-p0.bj.bcebos.com/Paddle-UnitTest-Model/PaddleDetection.tgz --no-check-certificate
    tar -zxf PaddleDetection.tgz
fi

cd -

bash $CASE_ROOT/pd-yolo-test.sh
bash $CASE_ROOT/pd-clas-test.sh
bash $CASE_ROOT/pd-rcnn-test.sh
bash $CASE_ROOT/pd-solo-test.sh
