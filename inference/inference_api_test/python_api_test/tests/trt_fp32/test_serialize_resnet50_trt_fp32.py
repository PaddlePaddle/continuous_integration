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
import os
import sys
import logging
import struct
import six
import shutil

import cv2
import wget
import pytest
import numpy as np

import paddle.inference as paddle_infer

sys.path.append("../..")
from src.img_preprocess import preprocess

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

def prepare_predictor(model_path, trt_max_batch_size=1):
    """
    prepare predictor
    """
    config = paddle_infer.Config(
        os.path.join(model_path, "__model__"),
        os.path.join(model_path, "__params__"))

    config.enable_memory_optim()
    config.enable_use_gpu(1000, 0)
    config.enable_tensorrt_engine(
        workspace_size=1 << 30,
        max_batch_size=trt_max_batch_size,
        min_subgraph_size=3,
        precision_mode=paddle_infer.PrecisionType.Float32,
        use_static=True,
        use_calib_mode=False)

    predictor = paddle_infer.create_predictor(config)
    return predictor


def run_inference(predictor, img):
    """
    run inference
    Args:
        predictor
        img
    Returns:
        results: numpy
    """
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results


@pytest.mark.p1
def test_trt_serialize_resnet50_bz1_pre():
    """
    test batch_size=1 serialize with resnet50 model
    Args:
        None
    Returns:
        None
    """
    if not os.environ.get("project_path"):
        logger.error("==== env project_path is not set ====")
        raise Exception("please export project_path=path/of/root_tests")
    model_root = os.path.join(
        os.environ.get("project_path"), "Data/python-model-infer")

    model_name = "ResNet50_pretrained"
    tmp_path = os.path.join(model_root, "classification")
    model_path = os.path.join(tmp_path, model_name, "model")
    test_img_url = "https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg"
    if not os.path.exists("./ILSVRC2012_val_00000247.jpeg"):
        wget.download(test_img_url, out="./")

    opt_cache_path = os.path.join(model_path, "_opt_cache")

    if os.path.exists(opt_cache_path):
        logger.warning("==== _opt_cache should be empty ====")
        logger.warning("==== _opt_cache will be remove ====")
        shutil.rmtree(opt_cache_path)
    assert os.path.exists(os.path.join(model_path, "_opt_cache")
                          ) == False, "_opt_cache is not empty before this test"

    files_before_serialize = os.listdir(model_path)
    logger.info("==== files_before_serialize: {} ====".format(
        files_before_serialize))

    predictor = prepare_predictor(model_path=model_path, trt_max_batch_size=1)

    img = cv2.imread('./ILSVRC2012_val_00000247.jpeg')
    img = preprocess(img)
    assert img[0].shape == (3, 224, 224), "input image should be 3, 224, 224"
    result = run_inference(predictor, [img])
    print("class index: ", np.argmax(result[0][0]))

    files_after_serialize = os.listdir(os.path.join(model_path, "_opt_cache"))
    logger.info("==== files_after_serialize: {} ====".format(
        files_after_serialize))

    assert len(files_after_serialize) == 1, "serialize file should be only one"


@pytest.mark.p1
def test_trt_serialize_resnet50_bzs():
    """
    test batch_size=1,2,4 serialize with resnet50 model
    Args:
        None
    Returns:
        None
    """
    if not os.environ.get("project_path"):
        logger.error("==== env project_path is not set ====")
        raise Exception("please export project_path=path/of/root_tests")
    model_root = os.path.join(
        os.environ.get("project_path"), "Data/python-model-infer")

    model_name = "ResNet50_pretrained"
    tmp_path = os.path.join(model_root, "classification")
    model_path = os.path.join(tmp_path, model_name, "model")
    test_img_url = "https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg"
    if not os.path.exists("./ILSVRC2012_val_00000247.jpeg"):
        wget.download(test_img_url, out="./")

    opt_cache_path = os.path.join(model_path, "_opt_cache")

    if os.path.exists(opt_cache_path):
        logger.warning("==== _opt_cache should be empty ====")
        logger.warning("==== _opt_cache will be remove ====")
        shutil.rmtree(opt_cache_path)
    assert os.path.exists(os.path.join(model_path, "_opt_cache")
                          ) == False, "_opt_cache is not empty before this test"

    files_before_serialize = os.listdir(model_path)
    logger.info("==== files_before_serialize: {} ====".format(
        files_before_serialize))

    # create 3 different predictor, should create 3 serialize hash files
    predictor = prepare_predictor(model_path=model_path, trt_max_batch_size=1)
    predictor = prepare_predictor(model_path=model_path, trt_max_batch_size=2)
    predictor = prepare_predictor(model_path=model_path, trt_max_batch_size=4)

    files_after_serialize = os.listdir(os.path.join(model_path, "_opt_cache"))
    logger.info("==== files_after_serialize: {} ====".format(
        files_after_serialize))

    assert len(files_after_serialize) == 3, "serialize file should be only one"
