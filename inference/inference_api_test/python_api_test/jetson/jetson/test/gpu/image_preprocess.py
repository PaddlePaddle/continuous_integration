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

import cv2
import numpy as np
from PIL import Image, ImageDraw


def resize(img, target_size):
    """
    resize to target size
    Args:
        img: img input(numpy)
        target_size: img size
    Returns:
        img: resize img
    """
    if not isinstance(img, np.ndarray):
        raise TypeError('image type is not numpy.')
    im_shape = img.shape
    im_scale_x = float(target_size) / float(im_shape[1])
    im_scale_y = float(target_size) / float(im_shape[0])
    img = cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y)
    return img


def normalize(img, mean, std):
    """
    normalize img
    Args:
        img: img input(numpy)
        mean: img mean
        std: img std
    Returns:
        img: normalize img
    """
    img = img / 255.0
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    img -= mean
    img /= std
    return img


def preprocess(img, img_size):
    """
    preprocess img
    Args:
        img_size: img size
    Returns:
        img: img add one axis
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = resize(img, img_size)
    img = img[:, :, ::-1].astype('float32')  # bgr -> rgb
    img = normalize(img, mean, std)
    img = img.transpose((2, 0, 1))  # hwc -> chw
    return img[np.newaxis, :]


def draw_bbox(img_name, result, threshold=0.5, save_name='res.jpg'):
    """
    draw bbox for detect img
    Args:
        img_name: img path
        result: box position
        threshold: threshold
        save_name: img save name
    Returns:
        None
    """
    img = Image.open(img_name).convert('RGB')
    draw = ImageDraw.Draw(img)
    for res in result:
        cat_id, score, bbox = res[0], res[1], res[2:]
        print(cat_id, score, bbox)
        if score < threshold:
            continue
        xmin, ymin, xmax, ymax = bbox
        draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                   (xmin, ymin)],
                  width=2,
                  fill=(255, 0, 0))
        print('category id is {}, bbox is {}'.format(cat_id, bbox))
    img.save(save_name, quality=95)


def seg_color(image):
    """
    color for seg img
    Args:
        img: seg out img
    Returns:
        result: seg img with color
    """
    label_colours = [
        [128, 64, 128],
        [244, 35, 231],
        [69, 69, 69]        # 0 = road, 1 = sidewalk, 2 = building
        ,
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153]        # 3 = wall, 4 = fence, 5 = pole
        ,
        [250, 170, 29],
        [219, 219, 0],
        [106, 142, 35]        # 6 = traffic light, 7 = traffic sign, 8 = vegetation
        ,
        [152, 250, 152],
        [69, 129, 180],
        [219, 19, 60]        # 9 = terrain, 10 = sky, 11 = person
        ,
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 69]        # 12 = rider, 13 = car, 14 = truck
        ,
        [0, 60, 100],
        [0, 79, 100],
        [0, 0, 230]        # 15 = bus, 16 = train, 17 = motocycle
        ,
        [119, 10, 32]
    ]
    result = []
    s = []
    for i in image.flatten():
        if i not in s:
            s.append(i)
        result.append(
            [label_colours[i][2], label_colours[i][1], label_colours[i][0]])
    result = np.array(result).reshape([image.shape[0], image.shape[1], 3])
    return result
