import argparse
import logging
import os
import sys
import time

import torch
import torchvision
import cv2

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    """[summary]
    Returns:
        [type]: [description]
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "vgg16-ssd", "mb1-ssd", "faster-rcnn", "resnet101", "mobilenetv2",
            "deeplabv3", "shufflenet", "vgg16", "googlenet", "squeezenet",
            "inception"
        ],
        help="network type")
    parser.add_argument(
        "--model_path",
        default="./MODEL.pth",
        type=str,
        help="model path, no need for torchvision models")
    parser.add_argument(
        "--trt_precision",
        type=str,
        default="fp32",
        help="trt precision, choice = ['fp32', 'fp16', 'int8']")
    parser.add_argument(
        "--image_shape",
        type=str,
        default="3,224,224",
        help="can only use for one input model(e.g. image classification)")

    parser.add_argument("--use_gpu", dest="use_gpu", action='store_true')
    parser.add_argument("--use_trt", dest="use_trt", action='store_true')
    parser.add_argument("--use_xla", dest="use_xla", action='store_true')

    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--warmup_times", type=int, default=5, help="warmup")
    parser.add_argument("--repeats", type=int, default=1000, help="repeats")

    return parser.parse_args()


def prepare_model(args):
    args = parse_args()
    if args.model_name == 'faster-rcnn':
        net = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
    elif args.model_name == 'resnet101':
        net = torchvision.models.resnet101(pretrained=True)
    elif args.model_name == 'vgg16':
        net = torchvision.models.vgg16(pretrained=True)
    elif args.model_name == 'googlenet':
        net = torchvision.models.googlenet(pretrained=True)
    elif args.model_name == 'shufflenet':
        net = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    elif args.model_name == 'mobilenetv2':
        net = torchvision.models.mobilenet_v2(pretrained=True)
    elif args.model_name == 'squeezenet':
        net = torchvision.models.squeezenet1_0(pretrained=True)
    elif args.model_name == 'inception':
        net = torchvision.models.inception_v3(pretrained=True)
    elif args.model_name == 'deeplabv3':
        net = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=True)
    else:
        raise Exception('net type [%s] invalid! \
                        \n please specify corret model_name' % args.model_name)

    logger.info("==== start jit scripts ====")
    traced_model = torch.jit.script(net)
    # traced_model = torch.jit.trace(net, [image_tensor])
    torch.jit.save(traced_model, "{}.jit.pt".format(args.model_name))
    logger.info("==== finish jit scripts ====")
    return net, traced_model


def prepare_input(args):
    channels = int(args.image_shape.split(',')[0])
    height = int(args.image_shape.split(',')[1])
    width = int(args.image_shape.split(',')[2])
    logger.info("channels: {0}, height: {1}, width: {2}".format(channels,
                                                                height, width))
    input_shape = (args.batch_size, height, width, channels)
    image_tensor = torch.randn(input_shape)
    return image_tensor


def forward_benchmark(args):
    image_tensor = prepare_input(args)
    logger.info("input image tensor shape : {}".format(image_tensor.shape))
    net, _ = prepare_model(args)
    # set running device on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    net = net.to(device)

    with torch.no_grad():
        # warm up
        for i in range(args.warmup_times):
            output = net(image_tensor)

        time1 = time.time()
        for i in range(args.repeats):
            output = net(image_tensor)
        time2 = time.time()
        total_inference_cost = (time2 - time1) * 1000  # total latency, ms
    return total_inference_cost


def trt_benchmark(args):
    import trtorch
    _, traced_model = prepare_model(args)

    if args.trt_precision == "fp32":
        image_tensor = prepare_input(args)
        trt_model = trtorch.compile(
            traced_model,
            {
                "input_shapes": [image_tensor.shape],
                "op_precision": torch.float32,  # Run with FP16
                "workspace_size": 1 << 20
            })
    elif args.trt_precision == "fp16":
        image_tensor = prepare_input(args)
        image_tensor = image_tensor.to(torch.half)
        trt_model = trtorch.compile(
            traced_model,
            {
                "input_shapes": [image_tensor.shape],
                "op_precision": torch.half,  # Run with FP16
                "workspace_size": 1 << 20
            })
        #"op_precision": torch.float32, # Run with FP32
    logger.info("input image tensor shape : {}".format(image_tensor.shape))

    for i in range(args.warmup_times):
        output = trt_model(image_tensor)

    time1 = time.time()
    for i in range(args.repeats):
        output = trt_model(image_tensor)
    time2 = time.time()
    total_inference_cost = (time2 - time1) * 1000  # total latency, ms
    return total_inference_cost


def summary_config(args, infer_time: float):
    """
    Args:
        args : input args
        infer_time : inference time
    """
    logger.info("----------------------- Model info ----------------------")
    logger.info("Model name: {0}, Model type: {1}".format(args.model_name,
                                                          "torch_model"))
    logger.info("----------------------- Data info -----------------------")
    logger.info("Batch size: {0}, Num of samples: {1}".format(args.batch_size,
                                                              args.repeats))
    logger.info("----------------------- Conf info -----------------------")
    logger.info("device: {0}".format("gpu" if args.use_gpu else "cpu"))
    if (args.use_gpu):
        logger.info("enable_tensorrt: {0}".format(args.use_trt))
        if (args.use_trt):
            logger.info("trt_precision: {0}".format(args.trt_precision))
    logger.info("----------------------- Perf info -----------------------")
    logger.info("Average latency(ms): {0}, QPS: {1}".format(
        infer_time / args.repeats, (args.repeats * args.batch_size) / (
            infer_time / 1000)))


def run_demo():
    """
    run_demo
    """
    args = parse_args()
    if args.use_trt:
        total_time = trt_benchmark(args)
    else:
        total_time = forward_benchmark(args)
    summary_config(args, total_time)


if __name__ == "__main__":
    run_demo()
