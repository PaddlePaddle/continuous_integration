# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
    """
    parse input argument
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument(
        "--model_path",
        default="./MODEL.pth",
        type=str,
        help="model path, no need for torchvision models")
    parser.add_argument(
        "--image_path",
        default="./image.jpg",
        type=str,
        help="input image path")
    parser.add_argument(
        "--input_shape", default="3,224,224", type=str, help="input data shape")
    parser.add_argument(
        "--batch_size", default=1, type=int, help="input data batch size")
    parser.add_argument(
        "--repeat_times", default=1000, type=int, help="repeat times")
    parser.add_argument(
        "--trt_precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16"],
        help="trt prediction precision")
    parser.add_argument("--use_trt", type=bool, help="use trt or not")
    return parser.parse_args()


def load_image(args):
    """
    load image
    """
    c, h, w = args.input_shape.split(',')
    if os.path.exists(args.image_path):
        orig_image = cv2.imread(args.image_path)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (h, w))
        image_tensor = torchvision.transforms.functional.to_tensor(image)
    else:
        image_tensor = torch.randn(args.batch_size, c, h, w)
    return image_tensor


def load_pretrain_model(args):
    """
    load_pretrain_model
    """
    if args.net_type == 'faster-rcnn':
        net = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
    elif args.net_type == 'resnet101':
        net = torchvision.models.resnet101(pretrained=True)
    elif args.net_type == 'vgg16':
        net = torchvision.models.vgg16(pretrained=True)
    elif args.net_type == 'googlenet':
        net = torchvision.models.googlenet(pretrained=True)
    elif args.net_type == 'shufflenet':
        net = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    elif args.net_type == 'mobilenetv2':
        net = torchvision.models.mobilenet_v2(pretrained=True)
    elif args.net_type == 'squeezenet':
        net = torchvision.models.squeezenet1_0(pretrained=True)
    elif args.net_type == 'inception':
        net = torchvision.models.inception_v3(pretrained=True)
    elif args.net_type == 'deeplabv3':
        net = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=True)
    elif args.net_type == 'vgg16-ssd':
        # ssd model
        net = None
    elif args.net_type == 'mb1-ssd':
        # ssd model
        net = None
    else:
        raise Exception('net type [%s] invalid! \
                        \n please specify corret net_type' % args.net_type)
        sys.exit(1)

    return net


def main():
    """
    main
    """
    args = parse_args()
    net = load_pretrain_model(args)
    net.eval()
    image_tensor = load_image(args)
    print(image_tensor.shape)

    # set running device on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    net = net.to(device)

    with torch.no_grad():
        # warm up 5 times
        for i in range(5):
            output = net(image_tensor)

        repeat_times = args.repeat_times
        t1 = time.time()
        for i in range(repeat_times):
            output = net(image_tensor)
        t2 = time.time()
        total_time = (t2 - t1) * 1000  # micro seconds
        average_time = total_time / repeat_times
    logger.info("total inference time is {} ms".format(total_time))
    logger.info("average inference time is {} ms".format(average_time))
    logger.info("prediction success")

    #save_net = torch.jit.script(net)
    #save_net.save("save_model.pt")
    model = net
    #traced_model = torch.jit.trace(model.to("cpu"), [image_tensor.to("cpu")])
    traced_model = torch.jit.trace(model, [image_tensor])
    #traced_model = torch.jit.trace(model.to("cpu"), [[torch.rand(3, 640, 640).to("cpu")]])
    #traced_model = torch.jit.script(model)
    torch.jit.save(traced_model, "{}.jit.pt".format(args.net_type))

    if use_trt:
        import trtorch
        image_shape = (1, 3, 224, 224)
        #image_shape = (1, 3, 640, 640)
        #image_shape = (1, 3, 769, 769)
        trt_model = trtorch.compile(
            traced_model,
            {
                "input_shapes": [image_shape],
                "op_precision": torch.half,  # Run with FP16
                "workspace_size": 1 << 20
            })
        #"op_precision": torch.float32, # Run with FP16
        #output = trt_model(image_tensor)
        #print(output)

        for i in range(5):
            #output = trt_model(image_tensor.to(torch.int8))
            #output = trt_model(image_tensor)
            output = trt_model(image_tensor.to(torch.half))

        repeat_times = 1000
        t1 = time.time()
        for i in range(repeat_times):
            #output = trt_model(image_tensor.to(torch.int8))
            output = trt_model(image_tensor.to(torch.half))
            #output = trt_model(image_tensor)
        t2 = time.time()
        total_time = (t2 - t1) * 1000  # micro seconds
        average_time = total_time / repeat_times
        logger.info("TRT total inference time is {} ms".format(total_time))
        logger.info("TRT average inference time is {} ms".format(average_time))
        logger.info("TRT prediction success")


if __name__ == "__main__":
    main()
