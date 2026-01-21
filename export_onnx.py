import torch
from model import get_resnet_mnist_model
import os

def export_to_onnx():
    """
    将训练好的 PyTorch 模型导出为 ONNX 格式。
    ONNX (Open Neural Network Exchange) 是一种开放的模型格式，可用于可视化 (Netron) 或在其他推理引擎 (TensorRT, ONNX Runtime) 中运行。
    """
    # 加载模型结构
    model = get_resnet_mnist_model()
    
    # 加载训练好的权重
    checkpoint_path = "checkpoints/resnet_mnist.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"Loaded weights from {checkpoint_path} (加载权重成功)")
    else:
        print("Warning: Checkpoint not found, exporting with random weights. (警告: 未找到权重，将使用随机权重导出)")
    
    model.eval() # 设置为评估模式
    
    # 创建一个虚拟输入 Tensor，匹配输入形状 (Batch, Channels, Height, Width)
    # MNIST 格式是 (1, 1, 28, 28)
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # 导出
    output_path = "checkpoints/resnet_mnist.onnx"
    torch.onnx.export(model, 
                      dummy_input, 
                      output_path, 
                      input_names=['input'], 
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}) # 支持动态 batch size
    
    print(f"Model exported to {output_path} (模型已导出)")
    print("You can view this file using Netron (https://netron.app) (您可以使用 Netron 查看此文件)")

if __name__ == "__main__":
    export_to_onnx()
