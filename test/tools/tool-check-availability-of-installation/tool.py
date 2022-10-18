# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

import os
import argparse
import logging
import numpy as np

import paddle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_item")
    args = parser.parse_args()
    return args


def _simple_network():
    """
    Define a simple network composed by a single linear layer.
    """
    paddle.enable_static()
    input = paddle.static.data(
        name="input", shape=[None, 2, 2], dtype="float32")
    weight = paddle.create_parameter(
        shape=[2, 3],
        dtype="float32",
        attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.1)))
    bias = paddle.create_parameter(shape=[3], dtype="float32")
    linear_out = paddle.nn.functional.linear(x=input, weight=weight, bias=bias)
    out = paddle.tensor.sum(linear_out)
    print(out)
    return input, out, weight


def _is_cuda_available():
    """
    Check whether CUDA is avaiable.
    """
    try:
        assert len(paddle.static.cuda_places()) > 0
        return True
    except Exception as e:
        logging.warning(
            "You are using GPU version PaddlePaddle, but there is no GPU "
            "detected on your machine. Maybe CUDA devices is not set properly."
            "\n Original Error is {}".format(e))
        return False


def _check_import():
    """
    Check import paddle.
    """
    try:
        import paddle
        return True
    except Exception as e:
        logging.warning('Import paddle failed: {}'.format(e))
        return False


if __name__ == '__main__':

    args = parse_args()
    if args.check_item == 'simple_network':
        _simple_network()

    if args.check_item == 'cuda_available':
        _is_cuda_available()

    if args.check_item == 'check_import':
        _check_import()
