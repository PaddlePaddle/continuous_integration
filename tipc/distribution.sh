
REPO=$1
CODE_BOS=$2
PADDLE_WHL=$3
CONFIG_FILE=$4
MODE=$5

wget $CODE_BOS --no-check-certificate

tar -zxf ${REPO}.tar.gz

cd Paddle
cp -r models/tutorials/mobilenetv3_prod/Step6 ./
rm -rf models
mv Step6 models

unlink /usr/local/bin/python
unlink /usr/local/bin/pip

ln -sf /usr/bin/python3.7 /usr/bin/python3 && \
ln -sf /usr/local/bin/pip3.7 /usr/local/bin/pip3 && \
ln -sf /usr/bin/python3 /usr/local/bin/python && \
ln -sf /usr/local/bin/pip3 /usr/local/bin/pip

export PATH=/home/cmake-3.16.0-Linux-x86_64/bin:/usr/local/bin:/usr/local/gcc-8.2/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH

python -V
pip -V

python -m pip install --retries 50 --upgrade pip -i https://mirror.baidu.com/pypi/simple
python -m pip config set global.index-url https://mirror.baidu.com/pypi/simple;
cd ./AutoLog
python -m pip install --retries 10 -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
python setup.py bdist_wheel
cd -
python -m pip install ./AutoLog/dist/*.whl
cd ./${REPO}
REPO_PATH=\`pwd\`
if [[ $REPO == "PaddleNLP" ]]; then
    cd tests
fi
if [[ $REPO == "PaddleGAN" ]]; then
    python -m pip install -v -e . #安装ppgan
fi
if [[ $REPO == "PaddleRec" ]]; then
    python -m pip install pgl
fi
python2 -m pip install --retries 10 pycrypto
python -m pip install --retries 10 Cython
python -m pip install --retries 10 distro
python -m pip install --retries 10 opencv-python
python -m pip install --retries 10 wget
python -m pip install --retries 10 pynvml
python -m pip install --retries 10 cup
python -m pip install --retries 10 pandas
python -m pip install --retries 10 openpyxl
python -m pip install --retries 10 psutil
python -m pip install --retries 10 GPUtil
python -m pip install --retries 10 paddleslim
#python -m pip install --retries 10 paddlenlp
python -m pip install --retries 10 attrdict
python -m pip install --retries 10 pyyaml
python -m pip install --retries 10 visualdl
python -c 'from visualdl import LogWriter'
python -m pip install --retries 10 -r requirements.txt
wget -q --no-proxy ${PADDLE_WHL}
python -m pip install ./\`basename ${PADDLE_WHL}\`

if [[ $REPO == "PaddleSeg" ]]; then
    pip install -e .
    python -m pip install --retries 50 scikit-image
    python -m pip install numba
    python -m pip install sklearn
fi
if [[ $REPO == "PaddleNLP" ]]; then
    python -m pip install --retries 10 paddlenlp
fi
cp \$REPO_PATH/../continuous_integration/tipc/upload.sh .

export FLAGS_selected_gpus=0,1
#bash test_tipc/prepare.sh test_tipc/configs/mobilenet_v3_small/train_fleet_infer_python.txt lite_train_lite_infer
#bash test_tipc/test_train_inference_python.sh test_tipc/configs/mobilenet_v3_small/train_fleet_infer_python.txt lite_train_lite_infer
bash test_tipc/prepare.sh ${CONFIG_FILE} ${MODE}
bash test_tipc/test_train_inference_python.sh ${CONFIG_FILE} ${MODE}
