import argparse
import os
import sys
import time
import logging

import numpy as np
import tensorflow as tf  # tf version should greater than 2.3.0

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.info("==== Tensorflow version: {} ====".format(tf.version.VERSION))
logger.info(
    "==== Tensorflow version greater than 2.3 can execute this codes ====")


def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, help="model save model input path")
    parser.add_argument(
        "--output_path", type=str, help="tf trt saved model output path")
    parser.add_argument(
        "--trt_precision",
        type=str,
        default="fp32",
        help="trt precision, choice = ['fp32', 'fp16', 'int8']")
    parser.add_argument(
        "--image_shape",
        type=str,
        default="3,224,224",
        help="can only use for one input model(e.g. image classification) to calibrate"
    )

    return parser.parse_args()


def save_trt_model(args):
    if not args.model_path:
        logger.error("==== no input model found ====")
        sys.exit(1)
    if not args.output_path:
        logger.error("==== no output_path found ====")
        sys.exit(1)

    input_saved_model_dir = args.model_path
    output_saved_model_dir = args.output_path
    logger.info("==== save trt model for inference ====")
    if args.trt_precision == "fp32":
        # convert model to trt fp32
        logger.info('Converting to TF-TRT FP32...')
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.FP32,
            max_workspace_size_bytes=8000000000)

        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=input_saved_model_dir,
            conversion_params=conversion_params)
        converter.convert()
        converter.save(output_saved_model_dir=output_saved_model_dir)
        logger.info('Done Converting to TF-TRT FP32')
    elif args.trt_precision == "fp16":
        logger.info('Converting to TF-TRT FP16...')
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.FP16,
            max_workspace_size_bytes=8000000000)
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=input_saved_model_dir,
            conversion_params=conversion_params)
        converter.convert()
        converter.save(output_saved_model_dir=output_saved_model_dir)
        logger.info('Done Converting to TF-TRT FP16')
    elif args.trt_precision == "int8":
        # convert model to trt int8
        logger.info('Converting to TF-TRT INT8...')
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.INT8,
            max_workspace_size_bytes=8000000000,
            use_calibration=True)
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=input_saved_model_dir,
            conversion_params=conversion_params)

        channels = int(args.image_shape.split(',')[0])
        height = int(args.image_shape.split(',')[1])
        width = int(args.image_shape.split(',')[2])
        logger.info("channels: {0}, height: {1}, width: {2}".format(
            channels, height, width))
        input_shape = (args.batch_size, height, width, channels)

        def calibration_input_fn(input_shape):
            """
            calib by all ones inputs
            """
            batched_input = tf.constant(np.ones(input_shape).astype("float"))
            batched_input = tf.cast(batched_input, dtype="float")
            yield (batched_input, )

        converter.convert(
            calibration_input_fn=calibration_input_fn(input_shape))
        converter.save(output_saved_model_dir=output_saved_model_dir)
        logger.info('Done Converting to TF-TRT INT8')
    else:
        logger.warn(
            "No TensorRT precision was input, will not convert TensorRT graph to saved model"
        )


if __name__ == "__main__":
    args = parse_args()
    save_trt_model(args)
