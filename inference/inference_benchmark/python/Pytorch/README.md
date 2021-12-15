# Torch python性能测试

## 测试依赖
```shell
import torch
import torchvision
import cv2
import trtorch
```
torch测试时，会默认打开了mkldnn优化

## 快速测试
快速执行单个测试

```shell
python torch_benchmark.py --model_name="resnet101" \
                          --input_shape"3,224,224" \
                          --batch_size=1 \
                          --repeat_times=1000
```

设置MKL线程为1
```shell
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```
