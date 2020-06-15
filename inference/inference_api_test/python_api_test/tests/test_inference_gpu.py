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
import argparse
import logging
import struct
import six

import nose
import numpy as np

sys.path.append("..")
from src.test_case import Predictor

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class TestModelInferenceGPU(object):
    """
    TestModelInferenceGPU
    Args:
    Return:
    """

    def __init__(self):
        """
        __init__
        """
        project_path = os.environ.get("project_path")
        self.model_root = os.path.join(project_path, "Data/python-model-infer")

    def check_data(self, result, expect, delta):
        """
        check result
        Args:
            result(list): list of result data
            expect(list): list of expect data
            delta(float): e.g. 0.001
        Return:
            None
        """
        logger.info("current comparison delta is : {0}".format(delta))
        nose.tools.assert_equal(
            len(expect), len(result), msg="output length not equal")
        for i in range(0, len(expect)):
            nose.tools.assert_almost_equal(expect[i], result[i], delta=delta)

    def get_infer_results(self, model_path, data_path):
        """
        get native and analysis infer results
        gpu
        Args:
            model_path(string): parent path of __model__ file
            data_path(string): path of data.json
        Return:
            res(numpy array): analysis cf outputs
            exp(numpy array): native cfg outputs
        """
        AnalysisPredictor = Predictor(
            model_path, predictor_mode="Analysis", config_type="gpu")
        res, ave_time = AnalysisPredictor.analysis_predict(data_path)
        logger.info(ave_time)

        try:
            NativePredictor = Predictor(
                model_path, predictor_mode="Native", config_type="gpu")
            exp, ave_time = NativePredictor.native_predict(data_path)
            logger.info(ave_time)
        except RuntimeError:
            logger.info("native prediction is out of gpu memory \
                         , use cpu native infer instead")
            NativePredictor = Predictor(
                model_path, predictor_mode="Native", config_type="cpu")
            exp, ave_time = NativePredictor.native_predict(data_path)
            logger.info(ave_time)

        nose.tools.assert_equal(
            len(exp), len(res), msg="num of output tensor not equal")
        return res, exp

    def test_inference_mobilenetv1_gpu(self):
        """
        Inference and check value
        mobilenetv1 gpu model
        Args:
            None
        Return:
            None
        """
        model_name = "MobileNetV1_pretrained"
        tmp_path = os.path.join(self.model_root, "classification")
        model_path = os.path.join(tmp_path, model_name, "model")
        data_path = os.path.join(tmp_path, model_name, "data/data.json")
        delta = 0.0001

        res, exp = self.get_infer_results(model_path, data_path)

        for i in range(len(res)):
            self.check_data(res[i].flatten(), exp[i].flatten(), delta)

    def test_inference_resnet50_gpu(self):
        """
        Inference and check value
        resnet50 gpu model
        Args:
            None
        Return:
            None
        """
        model_name = "ResNet50_pretrained"
        tmp_path = os.path.join(self.model_root, "classification")
        model_path = os.path.join(tmp_path, model_name, "model")
        data_path = os.path.join(tmp_path, model_name, "data/data.json")
        delta = 0.001

        res, exp = self.get_infer_results(model_path, data_path)

        for i in range(len(res)):
            self.check_data(res[i].flatten(), exp[i].flatten(), delta)

    def test_inference_seresnext50_gpu(self):
        """
        Inference and check value
        seresnext50 gpu model
        Args:
            None
        Return:
            None
        """
        model_name = "SE_ResNeXt50_32x4d_pretrained"
        tmp_path = os.path.join(self.model_root, "classification")
        model_path = os.path.join(tmp_path, model_name, "model")
        data_path = os.path.join(tmp_path, model_name, "data/data.json")
        delta = 0.0001

        res, exp = self.get_infer_results(model_path, data_path)

        for i in range(len(res)):
            self.check_data(res[i].flatten(), exp[i].flatten(), delta)

    def test_inference_xception41_gpu(self):
        """
        Inference and check value
        xception41 gpu model
        Args:
            None
        Return:
            None
        """
        model_name = "Xception_41_pretrained"
        tmp_path = os.path.join(self.model_root, "classification")
        model_path = os.path.join(tmp_path, model_name, "model")
        data_path = os.path.join(tmp_path, model_name, "data/data.json")
        delta = 0.0001

        res, exp = self.get_infer_results(model_path, data_path)

        for i in range(len(res)):
            self.check_data(res[i].flatten(), exp[i].flatten(), delta)

    def test_inference_blazeface_gpu(self):
        """
        Inference and check value
        blazeface gpu model
        Args:
            None
        Return:
            None
        """
        model_name = "blazeface_nas_128"
        tmp_path = os.path.join(self.model_root, "Detection")
        model_path = os.path.join(tmp_path, model_name, "model")
        data_path = os.path.join(tmp_path, model_name, "data/data.json")
        delta = 0.0001

        res, exp = self.get_infer_results(model_path, data_path)

        for i in range(len(res)):
            self.check_data(res[i].flatten(), exp[i].flatten(), delta)

    def test_inference_faster_rcnn_gpu(self):
        """
        Inference and check value
        faster_rcnn gpu model
        Args:
            None
        Return:
            None
        """
        model_name = "faster_rcnn_r50_1x"
        tmp_path = os.path.join(self.model_root, "Detection")
        model_path = os.path.join(tmp_path, model_name, "model")
        data_path = os.path.join(tmp_path, model_name, "data/data.json")
        delta = 0.001

        res, exp = self.get_infer_results(model_path, data_path)

        for i in range(len(res)):
            self.check_data(res[i].flatten(), exp[i].flatten(), delta)

    def test_inference_mask_rcnn_gpu(self):
        """
        Inference and check value
        mask_rcnn gpu model
        Args:
            None
        Return:
            None
        """
        model_name = "mask_rcnn_r50_1x"
        tmp_path = os.path.join(self.model_root, "Detection")
        model_path = os.path.join(tmp_path, model_name, "model")
        data_path = os.path.join(tmp_path, model_name, "data/data.json")
        delta = 0.001

        res, exp = self.get_infer_results(model_path, data_path)

        for i in range(len(res)):
            self.check_data(res[i].flatten(), exp[i].flatten(), delta)

    def test_inference_yolov3_gpu(self):
        """
        Inference and check value
        yolov3 gpu model
        Args:
            None
        Return:
            None
        """
        model_name = "yolov3_darknet"
        tmp_path = os.path.join(self.model_root, "Detection")
        model_path = os.path.join(tmp_path, model_name, "model")
        data_path = os.path.join(tmp_path, model_name, "data/data.json")
        delta = 0.001

        res, exp = self.get_infer_results(model_path, data_path)

        for i in range(len(res)):
            self.check_data(res[i].flatten(), exp[i].flatten(), delta)

    def test_inference_deeplabv3_gpu(self):
        """
        Inference and check value
        deeplabv3_mobilenetv2 gpu model
        Args:
            None
        Return:
            None
        """
        model_name = "deeplabv3_mobilenetv2"
        tmp_path = os.path.join(self.model_root, "segmentation")
        model_path = os.path.join(tmp_path, model_name, "model")
        data_path = os.path.join(tmp_path, model_name, "data/data.json")
        delta = 0.0001

        res, exp = self.get_infer_results(model_path, data_path)

        for i in range(len(res)):
            self.check_data(res[i].flatten(), exp[i].flatten(), delta)
