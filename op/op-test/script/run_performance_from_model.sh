#!/usr/bin/env bash

set -xe

nvidia-docker exec -e BUILD_VCS_NUMBER=${BUILD_VCS_NUMBER} -e TEST_ROOT_DIR=${TEST_ROOT_DIR} paddle_op_ce /bin/bash ./command_performance_from_model.sh

author=`git show ${BUILD_VCS_NUMBER} | head -2 | tail -1 | awk -F"Author: " '{print $2}'`
datetime=`git show ${BUILD_VCS_NUMBER} | head -3 | tail -1 | awk -F"Date: " '{print $2}' | awk '{printf("%s %s %s %s%s %s\n", $1, $2, $3, $4, $6, $5)}'`
timestamp=`date -d "$datetime" +%s`

cat ${TEST_ROOT_DIR}/ce/${BUILD_VCS_NUMBER}/performance_from_model/${BUILD_VCS_NUMBER} | while read line
do
    array=(${line//	/ })
    case_name=${array[0]}
    performance=${array[1]}
    performance_forward=${array[2]}
    if [ $performance = "-" ]; then
        performance=0
    fi
    if [ ${performance_forward} = "-" ]; then
        performance_forward=0
    fi
    docker exec mysql ./mysql -e "insert into paddle.op_performance(commitid, timestamp, author, case_name, performance, performance_forward) values('${BUILD_VCS_NUMBER}', $timestamp, '$author', '$case_name', $performance, ${performance_forward}) on duplicate key update performance=values(performance), performance_forward=values(performance_forward);"
done

#python alert.py ${BUILD_VCS_NUMBER} liuyiqun01
