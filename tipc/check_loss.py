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

import sys
import numpy as np

kv_dict = {}


def check_almost_equal(a, b, decimal):
    try:
        np.testing.assert_almost_equal(a, b, decimal)
        return True
    except:
        return False


with open(sys.argv[1]) as f:
    for l in f:
        l = l.rstrip('\r\n').split('[Train]')[1]
        kvs = l.split(']')[-1]
        label = ']'.join(l.split(']')[0:-1]) + ']'
        kv_list = kvs.replace(' ', '').split(',')
        for kv in kv_list:
            k, v = kv.split(':')
            kv_dict[label + k] = float(v)

with open(sys.argv[2]) as f:
    for l in f:
        l = l.rstrip('\r\n').split('[Train]')[1]
        kvs = l.split(']')[-1]
        label = ']'.join(l.split(']')[0:-1]) + ']'
        kv_list = kvs.replace(' ', '').split(',')
        for kv in kv_list:
            k, v = kv.split(':')
            if label + k in kv_dict:
                print('[CHECK]', '{}:{}'.format(sys.argv[3], label + k),
                      'expected: {}'.format(kv_dict[label + k]),
                      'actual: {}'.format(v),
                      check_almost_equal(float(v), kv_dict[label + k], 2))
            else:
                print('[CHECK]', '{}:{}'.format(sys.argv[3], label + k),
                      'NOT FOUND')
