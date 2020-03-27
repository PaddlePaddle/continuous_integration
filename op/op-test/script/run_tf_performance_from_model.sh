#!/usr/bin/env bash

set -xe

case_pattern=$1

BUILD_VCS_NUMBER=tf_from_model

nvidia-docker exec -e BUILD_VCS_NUMBER=${BUILD_VCS_NUMBER} -e TEST_ROOT_DIR=${TEST_ROOT_DIR} paddle_op_ce /bin/bash ./command_tf_performance_from_model.sh $case_pattern

cat ${TEST_ROOT_DIR}/ce/${BUILD_VCS_NUMBER}/${BUILD_VCS_NUMBER} | while read line
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
    docker exec mysql ./mysql -e "insert into paddle.case_meta(case_name, tf_performance, tf_performance_forward) values('${case_name}', $performance, ${performance_forward}) on duplicate key update tf_performance=values(tf_performance), tf_performance_forward=values(tf_performance_forward);"
done
