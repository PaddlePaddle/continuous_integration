set -ex

mkdir -p build
cd build

cmake ../src
make -j4
cd ..

export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib64

# XLA env
# export TF_XLA_FLAGS="--tf_xla_cpu_global_jit"
# export TF_XLA_FLAGS="--tf_xla_auto_jit=2" # GPU
# export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" # CPU

./build/clas_benchmark --model_path=./mobilenet_v2_1.0_224_frozen.pb --repeats=1