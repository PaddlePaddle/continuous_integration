#!/usr/bin/env bash
set -eo pipefail


# clone models repo from gitbub
if [ -e models ]
then
	sudo /bin/rm -rf models
fi
#git clone https://github.com/PaddlePaddle/models.git
git clone -b zytest https://github.com/zhengya01/models.git

# clone Paddle repo from gitbub
if [ -e Paddle ]
then
	sudo /bin/rm -rf Paddle
fi
git clone https://github.com/PaddlePaddle/Paddle.git

# get models which changed
sh  commit.sh `pwd`

# run models in docker
./run_test `pwd`
