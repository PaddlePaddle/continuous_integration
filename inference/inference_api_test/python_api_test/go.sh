#!/bin/bash
set -x
project_path=`pwd`
export project_path
echo -e "\033[33m project_path is : ${project_path} \033[0m"
cd ${project_path}
test_cases=`find ./tests -name "test_v1*py" | sort`
cases="${test_cases}"
ignore=""
# download Data
if [ -d "Data" ];then rm -rf Data
fi
# download data with numpy
wget --no-proxy -q https://sys-p0.bj.bcebos.com/inference/Data.tgz --no-check-certificate
tar -xvf Data.tgz
if [ -d "result" ];then rm -rf report
fi
report_path="${project_path}/report"
mkdir -p ${report_path}
for file in ${cases}
do
    test_case=`basename ${file}`
    test_case_path=`dirname ${file}`
    cd ${test_case_path}
    echo -e "\033[33m ====> ${test_case} case start \033[0m"
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        # python -m nose ${test_case} --with-allure --logdir=${report_path}
        python -m pytest -v --disable-warnings  ${test_case}
    fi
    echo -e "\033[33m ====> ${test_case} case finish \033[0m"
    echo " "
    cd -
done
