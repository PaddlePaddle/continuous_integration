# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

"""
Cases for inference, ResNet50Model.

"""
import random
import numpy as np
import paddle
import paddle.fluid as fluid
import api_infer.InferAPI as InferAPI
import cv2


class ResNet50Model(InferAPI.InferApiTest):
    """
    ResNet50 Model configurations and it's functions.

    Attributes:
        inherit from InferApiTest class
    """

    def __init__(self):
        super(ResNet50Model, self).__init__("resnet50")
        pass

    def img_reader(self, im_path, size, mean, std):
        """
        Return image data
        Args:
            im_path(string|.jpg):
            size(int):
            mean(float):
            std(float):
        Returns:
            out_img: 
        """
        im = cv2.imread(im_path).astype('float32')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w, _ = im.shape
        im_scale_x = size / float(w)
        im_scale_y = size / float(h)
        out_img = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=cv2.INTER_CUBIC)
        mean = np.array(mean).reshape((1, 1, -1))
        std = np.array(std).reshape((1, 1, -1))
        out_img = (out_img / 255.0 - mean) / std
        out_img = out_img.transpose((2, 0, 1))
        return out_img

    def load_fake_data(self, batch_size=1, channels=3, height=224, width=224):
        """
        load fake data(all one data)
        Args:
            batch_size(int): 
            channels(int): 
            height(int):
            width(int):
        Returns:
            input_value: 
        """
        input_num = channels * height * width * batch_size
        input_data = [[1 for x in range(input_num)]]

        sum_i = 0
        for i in range(input_num):
            input_data[0][i] = 1
            sum_i += input_data[0][i]
        print("sum_i: {}".format(sum_i))

        the_data = []
        for data in input_data:
            the_data += data
        the_data = np.array(the_data).astype(np.float32)
        # notice the input data must be float

        input_tensor = fluid.core.PaddleTensor()
        input_tensor.shape = [batch_size, channels, height, width]
        input_tensor.data = fluid.core.PaddleBuf(the_data.tolist())
        input_value = [input_tensor]
        return input_value

    def load_random_data(self, batch_size=1, channels=3, height=224, 
            width=224):
        """
        load random data
        Args:
            batch_size(int): 
            channels(int): 
            height(int):
            width(int):
        Returns:
            input_value: 
        """
        input_num = channels * height * width * batch_size
        input_data = np.array(
            [random.randint(-200, 100)
             for x in range(input_num)]).astype(np.float32)
        input_tensor = fluid.core.PaddleTensor()
        input_tensor.shape = [batch_size, channels, height, width]
        input_tensor.data = fluid.core.PaddleBuf(input_data.tolist())
        input_tensor.dtype = fluid.core.PaddleDType.FLOAT32
        input_value = [input_tensor]
        return input_value

    def load_real_data(self,
                       img_path,
                       batch_size=1,
                       channels=3,
                       height=224,
                       width=224):
        """
        load real data
        Args:
            batch_size(int): 
            channels(int): 
            height(int):
            width(int):
            img_path(string|.jpg): "/path/of/resnet50/test_image.jpg"

        Returns:
            input_value: 
        """
        out_img = self.img_reader(img_path, width, 0.0, 1.0)
        input_img = np.array([out_img] * batch_size, dtype="float32")
        image = fluid.core.PaddleTensor(data=input_img.astype('float32'))

        input_value = [image]
        return input_value
