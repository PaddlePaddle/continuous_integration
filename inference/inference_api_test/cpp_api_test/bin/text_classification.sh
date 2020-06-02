#!/usr/bin/env bash
set -eo pipefail
echo "Test starting... text_classification thread 4 batch_size 4 use gpu"
$OUTPUT_BIN/test_text_classification --use_gpu=true \
                                     --infer_model=$DATA_ROOT/cpp-model-infer/text_classification/model \
                                     --infer_data=$DATA_ROOT/cpp-model-infer/text_classification/data/data.txt \
                                     --batch_size=4 \
                                     --num_threads=4 \
                                     --repeat=3 \
                                     --gtest_output=xml:test_text_classification_gpu.xml

echo "Test starting... text_classification thread 4 batch_size 4 use cpu"
$OUTPUT_BIN/test_text_classification --use_gpu=false \
                                     --infer_model=$DATA_ROOT/cpp-model-infer/text_classification/model \
                                     --infer_data=$DATA_ROOT/cpp-model-infer/text_classification/data/data.txt \
                                     --batch_size=4 \
                                     --num_threads=4 \
                                     --repeat=3 \
                                     --gtest_output=xml:test_text_classification_cpu.xml
