#### 1.工具介绍

二分定位工具用于对日常发现的问题进行精确PR级定位

#### 2.工具使用

以PaddleTest中的case为例，介绍工具的使用方式

先将binary_search_pr.py和run.sh文件拷贝到demo运行目录中

```shell
#使用编译缓存，提升编译速度
yum install ccache -y
mkdir -p /root/.ccache/coverage
export CCACHE_DIR=/root/.ccache/coverage
export CCACHE_MAXSIZE=30G
#使用二分查找pr工具
nohup python binary_search_pr.py \
--start_commit=36d76840acf06b6b7f95803001dce9952cc43b77 \
--end_commit=ce5e119696084cf8836a182df1b814c2dd80a256 \
--command='bash run.sh' \
--cmake_command='cmake .. -DON_INFER=ON -DWITH_PYTHON=ON -DPY_VERSION=3.8 -DCMAKE_BUILD_TYPE=Release  \
                -DWITH_MKL=ON -DWITH_AVX=OFF -DWITH_MKLDNN=ON -DWITH_GPU=ON -DWITH_TENSORRT=ON  \
                -DTENSORRT_ROOT=/usr/local/TensorRT-8.0.3.4 -DWITH_TESTING=OFF -DWITH_INFERENCE_API_TEST=OFF  \
                -DWITH_DISTRIBUTE=OFF -DWITH_STRIP=ON -DWITH_CINN=OFF -DWITH_ONNXRUNTIME=OFF  \
                -DCUDA_ARCH_NAME=Auto' 2>&1 &
#start_commit是查找开始的commit编号，其运行结果是正常的；end_commit是查找结束的commit编号，其运行结果是异常的。
#cmake_command是编译预测库使用的cmake命令，视排查的问题而定
```

需要注意的问题：

1. 根据排查的具体问题，可以调整cmake_command和run.sh的内容，需要保证在有问题的commit下运行run.sh时退出码不为0