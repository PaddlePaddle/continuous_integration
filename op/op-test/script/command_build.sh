#!/usr/bin/env bash

set -xe

alias rm=rm

# prepare
rm -rf ./build
rm -f /paddle/build/opt/paddle/share/wheels/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl

# build
cd /paddle/paddle/scripts
./paddle_build.sh build_only

# install paddle
pip uninstall -y paddlepaddle-gpu
pip install /paddle/build/opt/paddle/share/wheels/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
