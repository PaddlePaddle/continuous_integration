#!/bin/sh


ROOT_PATH=$1

if [ -e data ]
then
   mv data data.bak
fi
if [ ! -e data.tgz ]
then
    wget https://sys-p0.bj.bcebos.com/models/PaddleNLP/seq2seq/seq2seq/data.tgz --no-check-certificate
fi
tar -zxf data.tgz
