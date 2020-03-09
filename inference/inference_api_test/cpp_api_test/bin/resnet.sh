#!/usr/bin/env bash
set -eo pipefail

echo "Test starting... resnet50 thread 4 batch_size 4 use gpu fluid"
$OUTPUT_BIN/test_resnet50 --infer_model=$DATA_ROOT/c++/resnet50/model --infer_data=$DATA_ROOT/c++/resnet50/data/data.bin --batch_size=4 --num_threads=4 --repeat=3 --use_gpu=true --gtest_output=xml:test_resnet50_gpu_fluid.xml

echo "Test starting... resnet50 thread 4 batch_size 4 use cpu mkldnn"
$OUTPUT_BIN/test_resnet50 --infer_model=$DATA_ROOT/c++/resnet50/model --infer_data=$DATA_ROOT/c++/resnet50/data/data.bin --batch_size=4 --num_threads=4 --repeat=3 --use_gpu=false --use_mkldnn=true --gtest_output=xml:test_resnet50_cpu_mkldnn.xml

echo "Test starting... resnet50 thread 4 batch_size 4 use cpu mkldnn disable_mkldnn_fc"
$OUTPUT_BIN/test_resnet50 --infer_model=$DATA_ROOT/c++/resnet50/model --infer_data=$DATA_ROOT/c++/resnet50/data/data.bin --batch_size=4 --num_threads=4 --repeat=3 --use_gpu=false --use_mkldnn=true --disable_mkldnn_fc=true --gtest_output=xml:test_resnet50_cpu_mkldnn_disable_mkldnn_fc.xml


