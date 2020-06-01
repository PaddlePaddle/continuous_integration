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
if [ ! -f c++/resnet50/model/__model__ ]; then
    echo "==== Download data and models ===="
    wget --no-proxy https://sys-p0.bj.bcebos.com/inference/c++-infer.tgz --no-check-certificate
    tar -zxf c++-infer.tgz
fi

if [ ! -f  cpp-model-infer/bert_emb128/model/__model__ ]; then
    echo "==== Download bert, ocr, text data and models ===="
    wget --no-proxy https://sys-p0.bj.bcebos.com/inference/cpp-model-infer.tgz --no-check-certificate
    tar -zxf cpp-model-infer.tgz
fi

cd -

bash $CASE_ROOT/resnet.sh

bash $CASE_ROOT/ocr.sh
bash $CASE_ROOT/text_classification.sh
bash $CASE_ROOT/bert.sh
