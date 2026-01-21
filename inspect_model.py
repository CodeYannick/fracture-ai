import torch
import os

def inspect_pth(file_path):
    """
    检查 .pth 文件的内容（模型权重）。
    打印每层的名称和对应的张量形状。
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    try:
        # 加载状态字典 (State Dictionary)
        state_dict = torch.load(file_path, map_location='cpu')
        
        print("\nModel State Dictionary Structure (模型状态字典结构):")
        print("-" * 60)
        print(f"{'Layer Name (层名称)':<40} | {'Shape (形状)':<20}")
        print("-" * 60)
        
        total_params = 0
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                shape_str = str(list(value.shape))
                print(f"{key:<40} | {shape_str:<20}")
                total_params += value.numel()
        
        print("-" * 60)
        print(f"Total Parameters (总参数量): {total_params:,}")
        
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    inspect_pth("checkpoints/resnet_mnist.pth")
