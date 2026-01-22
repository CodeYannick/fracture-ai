import torch
import torch.nn as nn
from torchvision import models

def get_resnet_mnist_model(num_classes=10):
    """
    获取适用于 MNIST 数据集的 ResNet18 模型。
    因为 MNIST 是单通道（灰度）图像，我们需要修改标准 ResNet18 的输入层和输出层。
    
    Args:
        num_classes (int): 输出类别的数量。默认为 10 (MNIST 数字 0-9)。
                          如果要识别字母，可以改为 62 (10数字 + 26大写 + 26小写) 或其他数量。
    """
    # 加载 ResNet18 模型
    # 对于像 MNIST 这样简单的数据集，我们不需要预训练权重 (weights=None)。
    # 使用预训练权重可能会加速收敛，但这里为了演示从头训练，我们不使用它。
    model = models.resnet18(weights=None)
    
    # 修改第一个卷积层
    # 原始 ResNet18: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # MNIST 图像是 28x28 灰度图 (1 个通道)。
    # 为了避免在小图像上进行过激的下采样（导致特征图过早变得太小），我们可以调整 kernel_size 和 stride。
    # 标准 ResNet: 7x7 卷积 stride 2 -> 14x14 输出。
    # 我们可以将其改为 3x3 卷积 stride 1 来保留更多的空间信息。
    # 但为了保持最简单的修改（只改通道数），这里我们只修改输入通道数为 1。
    # 结构流: 28x28 -> (stride 2) 14x14 -> (maxpool stride 2) 7x7 ... 最后变成 1x1。
    
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # 修改最后的全连接层 (Fully Connected Layer)
    # 原始: nn.Linear(512, 1000) (ImageNet 有 1000 类)
    # MNIST 只有 10 个类别 (数字 0-9)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
