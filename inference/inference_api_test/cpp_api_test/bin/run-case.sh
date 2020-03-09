#!/usr/bin/env bash

set -eo pipefail

ROOT=`dirname "$0"`
ROOT=`cd "$ROOT/.."; pwd`
export OUTPUT=$ROOT/output
export OUTPUT_BIN=$ROOT/build
export DATA_ROOT=$ROOT/Data
export TOOLS_ROOT=$ROOT/tools
export CASE_ROOT=$ROOT/bin

bash $CASE_ROOT/resnet.sh

