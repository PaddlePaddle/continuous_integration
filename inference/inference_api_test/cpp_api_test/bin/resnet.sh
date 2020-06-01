#!/usr/bin/env bash
set -eo pipefail

echo "Test starting... resnet50 thread 4 batch_size 4 GPU"
$OUTPUT_BIN/test_resnet50 --infer_model=$DATA_ROOT/c++/resnet50/model \
                          --infer_data=$DATA_ROOT/c++/resnet50/data/data.bin \
                          --batch_size=4 --num_threads=4 \
                          --repeat=3 --use_gpu=true \
                          --gtest_output=xml:test_resnet50_fluid_gpu.xml

echo "Test starting... resnet50 thread 4 batch_size 4 CPU"
$OUTPUT_BIN/test_resnet50 --infer_model=$DATA_ROOT/c++/resnet50/model \
                          --infer_data=$DATA_ROOT/c++/resnet50/data/data.bin \
                          --batch_size=4 --num_threads=4 \
                          --repeat=3 --use_gpu=false \
                          --gtest_output=xml:test_resnet50_fluid_cpu.xml
