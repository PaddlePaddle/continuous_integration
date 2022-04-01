#!/usr/bin/env bash
set -eo pipefail

echo "Test starting... bert thread 4 batch_size 4 use gpu fluid"
$OUTPUT_BIN/test_bert --infer_model=$DATA_ROOT/cpp-model-infer/bert_emb128/model \
                      --infer_data=$DATA_ROOT/cpp-model-infer/bert_emb128/data/data.txt \
                      --repeat=3 --num_threads=4 \
                      --use_gpu=true \
                      --batch_size=4 --gtest_output=xml:test_bert_gpu.xml

echo "Test starting... bert thread 4 batch_size 4 use cpu fluid"
$OUTPUT_BIN/test_bert --infer_model=$DATA_ROOT/cpp-model-infer/bert_emb128/model \
                      --infer_data=$DATA_ROOT/cpp-model-infer/bert_emb128/data/data.txt \
                      --repeat=3 --num_threads=4 \
                      --use_gpu=false \
                      --batch_size=4 --gtest_output=xml:test_bert_cpu.xml
