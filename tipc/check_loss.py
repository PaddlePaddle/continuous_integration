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

kv_dict = {}

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
                      'actual: {}'.format(v), float(v) == kv_dict[label + k])
            else:
                print('[CHECK]', '{}:{}'.format(sys.argv[3], label + k),
                      'NOT FOUND')
