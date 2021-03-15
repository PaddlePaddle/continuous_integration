import argparse
import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType
import os
def main(ir_optim=True):
    args = parse_args()
    config = Config(args.model_file)
    config.enable_use_gpu(1000, 0)
    config.switch_ir_optim(ir_optim)
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    config.enable_memory_optim()
    predictor = create_predictor(config)

    input_names = predictor.get_input_names()
    input_handle0 = predictor.get_input_handle(input_names[0])
    input_handle1 = predictor.get_input_handle(input_names[1])
    input_handle2 = predictor.get_input_handle(input_names[2])
    input_handle3 = predictor.get_input_handle(input_names[3])
    input_handle4 = predictor.get_input_handle(input_names[4])
    input_handle5 = predictor.get_input_handle(input_names[5])
    input_handle6 = predictor.get_input_handle(input_names[6])
    input_handle7 = predictor.get_input_handle(input_names[7])
    
    np.random.seed(0)
    fake_input0 = np.random.randn(args.batch_size, 128, 1).astype("int64")
    input_handle0.reshape([1, 128, 1])
    input_handle0.copy_from_cpu(fake_input0)

    np.random.seed(1)
    fake_input1 = np.random.randn(args.batch_size, 128, 1).astype("int64")
    input_handle1.reshape([1, 128, 1])
    input_handle1.copy_from_cpu(fake_input1)
    
    np.random.seed(2)
    fake_input2 = np.random.randn(args.batch_size, 128, 1).astype("float32")
    input_handle2.reshape([1, 128, 1])
    input_handle2.copy_from_cpu(fake_input2)

    np.random.seed(3)
    fake_input3 = np.random.randn(args.batch_size, 128, 1).astype("int64")
    input_handle3.reshape([1, 128, 1])
    input_handle3.copy_from_cpu(fake_input3)
    
    np.random.seed(4)
    fake_input4 = np.random.randn(args.batch_size, 128, 1).astype("int64")
    input_handle4.reshape([1, 128, 1])
    input_handle4.copy_from_cpu(fake_input4)
    
    np.random.seed(5)
    fake_input5 = np.random.randn(args.batch_size, 128, 1).astype("int64")
    input_handle5.reshape([1, 128, 1])
    input_handle5.copy_from_cpu(fake_input5)

    np.random.seed(6)
    fake_input6 = np.random.randn(args.batch_size, 128, 1).astype("float32")
    input_handle6.reshape([1, 128, 1])
    input_handle6.copy_from_cpu(fake_input6)

    np.random.seed(7)
    fake_input7 = np.random.randn(args.batch_size, 128, 1).astype("float32")
    input_handle7.reshape([1, 128, 1])
    input_handle7.copy_from_cpu(fake_input7)

    
    predictor.run()
    
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()
    return output_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file",default='../../Data/hub-ernie/hub-ernie-model' ,type=str, help="model filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    return parser.parse_args()

if __name__ == "__main__":
    try:
        ir_optim_data = main(ir_optim=False)
        data = main(ir_optim=True)
    except:
        os.system('echo "GPU hub-ernie inference error" >> log.txt')
    else:
        max_diff = np.max(np.abs(data - ir_optim_data))
        if max_diff < 1e-6:
            print('GPU hub-ernie ok')
        else:
            print('GPU hub-ernie failed, diff is ', max_diff)
            os.system('echo "GPU hub-ernie failed,diff is "' + str(max_diff) + '>> log.txt')
