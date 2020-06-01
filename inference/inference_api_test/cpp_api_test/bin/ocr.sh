#!/usr/bin/env bash
set -eo pipefail

echo "Test starting... ocr thread 4 batch_size 4 use gpu fluid"
$OUTPUT_BIN/test_ocr --batch_size=4 \
                     --use_gpu=true \
                     --infer_model=$DATA_ROOT/cpp-model-infer/ocr/model \
                     --infer_data=$DATA_ROOT/cpp-model-infer/ocr/data/data.txt \
                     --refer_result=$DATA_ROOT/cpp-model-infer/ocr/data/result.txt \
                     --repeat=3 --num_threads=4 \
                     --gtest_output=xml:test_ocr_gpu.xml

echo "Test starting... ocr thread 4 batch_size 4 use cpu fluid"
$OUTPUT_BIN/test_ocr --batch_size=4 \
                     --use_gpu=false \
                     --infer_model=$DATA_ROOT/cpp-model-infer/ocr/model \
                     --infer_data=$DATA_ROOT/cpp-model-infer/ocr/data/data.txt \
                     --refer_result=$DATA_ROOT/cpp-model-infer/ocr/data/result.txt \
                     --repeat=3 --num_threads=4 \
                     --gtest_output=xml:test_ocr_cpu.xml
