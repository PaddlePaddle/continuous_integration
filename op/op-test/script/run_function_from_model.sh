#!/usr/bin/env bash

set -xe

WORKING_DIR=`pwd`
nvidia-docker exec -e BUILD_VCS_NUMBER=${BUILD_VCS_NUMBER} -e TEST_ROOT_DIR=${TEST_ROOT_DIR} paddle_op_ce /bin/bash ./command_function_from_model.sh $WORKING_DIR
