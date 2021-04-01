project_path=$(cd "$(dirname "$0")";pwd)
echo -e "\033[33m project_path is : ${project_path} \033[0m"

if [ ! -d "Data" ]; then
  wget --no-proxy --no-check-certificate -q https://sys-p0.bj.bcebos.com/inference/Data.tgz
  tar -xf Data.tgz
fi

ModelCase["gpu"]="test_resnet50.py \
                  test_ssd_mobilenet.py \
                  test_ttfnet.py \
                  test_ppyolo_r50vd.py \
                  test_ssd_vgg16.py \
                  test_ssd_mobilenet.py \
                  test_deeplabv3p_resnet50.py"

export project_path
for config in "gpu"
do
    cd ${project_path}/test/${config}
    for file in ${ModelCase[${config}]}
    do  
        echo " "
        echo -e "\033[33m ====> ${file} case start \033[0m"
        python3.8 -m pytest -m p0 ${file}
        echo -e "\033[33m ====> ${file} case finish \033[0m"
    done
done
