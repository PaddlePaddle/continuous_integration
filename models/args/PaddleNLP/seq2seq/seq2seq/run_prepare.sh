#!/bin/sh

ROOT_PATH=$1

rm -rf data
#ln -s /ssd3/models_test/models_args/PaddleNLP/PaddleTextGEN/seq2seq/data data
ln -s ${ROOT_PATH}/data/PaddleNLP/seq2seq/seq2seq/data data
