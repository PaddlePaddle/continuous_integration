#!/usr/bin/env bash
set -eo pipefail


# clone models repo from gitbub
if [ -e models ]
then
	sudo /bin/rm -rf models
fi
git clone https://github.com/PaddlePaddle/models.git

# clone Paddle repo from gitbub
if [ -e Paddle ]
then
	sudo /bin/rm -rf Paddle
fi
git clone https://github.com/PaddlePaddle/Paddle.git

# get models which changed
sh  commit.sh

# run models in docker
./run_test `pwd`
