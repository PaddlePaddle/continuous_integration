import os
import time
import threading

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

import tf_helper as helper

tf.disable_v2_behavior()

def Inference(args, config) -> int:
    input_node = args.input_node # "import/inputs:0"
    output_node = args.output_node # "resnet_v1_101/predictions/Reshape_1:0"
    model_path = args.model_path  # "resnet_v1_101.pb"

    # prepare input data
    channels = int(args.image_shape.split(',')[0])
    height = int(args.image_shape.split(',')[1])
    width = int(args.image_shape.split(',')[2])
    helper.logger.info("channels: {0}, height: {1}, width: {2}".format(channels,
                                                                       height,
                                                                       width))
    input_data = np.zeros((args.batch_size, height, width, channels))

    # set runtime devices
    if args.use_gpu:
        place = "/gpu:0"  
    else:
        place = "/cpu:0"  # will still load cuda lib

    trt_precision_map = {"fp32" : trt.TrtPrecisionMode.FP32,
                         "fp16" : trt.TrtPrecisionMode.FP16,
                         "int8" : trt.TrtPrecisionMode.INT8}

    # with tf.device(place):
    with tf.Session(config=config) as sess:
        # First deserialize your frozen graph:
        with tf.io.gfile.GFile(model_path, 'rb') as f:
            frozen_graph = tf.GraphDef()
            frozen_graph.ParseFromString(f.read())
        # Now you can create a TensorRT inference graph from your
        use_calib = True if args.trt_precision == "int8" else False
        if args.use_trt:
            converter = trt.TrtGraphConverter(
                input_graph_def=frozen_graph,
                max_batch_size=args.batch_size,
                precision_mode=trt_precision_map[args.trt_precision],
                use_calibration=use_calib,
                nodes_blacklist=[output_node])
            trt_graph = converter.convert()
            output_node = tf.import_graph_def(
                trt_graph,
                return_elements=[output_node])
        else:
            output_node = tf.import_graph_def(
                frozen_graph,
                return_elements=[output_node])

        for i in range(args.warmup_times):
            sess.run(output_node, {input_node : input_data})

        time1 = time.time()
        for i in range(args.repeats):
            sess.run(output_node, {input_node : input_data})
        time2 = time.time()
        total_inference_cost = (time2 - time1) * 1000  # total latency, ms

    return total_inference_cost

def run_demo():
    """
    run_demo
    """
    from py_mem import record_from_pid
    record_thread = threading.Thread(target=record_from_pid, args=(os.getpid(), ))
    record_thread.setDaemon(True)
    record_thread.start()
    
    args = helper.parse_args()
    config = helper.prepare_config(args)
    total_time = Inference(args, config)
    helper.summary_config(args, total_time)

if __name__ == "__main__":
    run_demo()