#!/bin/bash
set -ex
#set param
#paddle_version=develop
paddle_father_path=$PWD
# if build with TensorRT, pass TRT root in
TENSORRT_ROOT=""
if [ $# -ge 1 ]; then
  TENSORRT_ROOT="$1"
fi
#set environment
cd ${paddle_father_path}
export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.7.0/bin/python3.7 -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.7.0/include/python3.7m -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.7.0/lib/libpython3.7m.so"
export PYTHON_ABI="cp37-cp37m"
export PY_VERSION=3.7
yum -y install python-devel
pip install numpy protobuf wheel
if [ $? -ne 0 ];then
    echo "install numpy failed!";exit;
fi

if [ -d "Paddle" ];then rm -rf Paddle
fi
#git clone https://github.com/PaddlePaddle/Paddle.git
#cd Paddle

#git checkout release/${paddle_version}
if [ -d "build" ];then rm -rf build
fi
mkdir build
cd build
cmake .. -DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.7.0/bin/python3.7 \
         -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.7.0/include/python3.7m \
         -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.7.0/lib/libpython3.7m.so \
         -DWITH_FLUID_ONLY=ON \
         -DWITH_GPU=ON \
         -DWITH_TESTING=OFF \
         -DCMAKE_BUILD_TYPE=Release \
         -DON_INFER=ON \
         -DWITH_INFERENCE_API_TEST=OFF \
         -DWITH_MKL=ON \
         -DWITH_MKLDNN=ON \
         -DTENSORRT_ROOT=${TENSORRT_ROOT}
if [ $? -ne 0 ];then
    echo -e "\033[33m cmake failed! \033[0m";exit;
    echo -e "\033[33m cmake successfully! \033[0m";
fi
make -j$(nproc)
if [ $? -ne 0 ];then
    echo -e "\033[33m make paddle failed! \033[0m";exit;
    echo -e "\033[33m make paddle successfully! \033[0m";
fi
