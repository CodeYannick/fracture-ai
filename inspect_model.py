import torch
import os

def inspect_pth(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    try:
        # Load the state dictionary
        state_dict = torch.load(file_path, map_location='cpu')
        
        print("\nModel State Dictionary Structure:")
        print("-" * 60)
        print(f"{'Layer Name':<40} | {'Shape':<20}")
        print("-" * 60)
        
        total_params = 0
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                shape_str = str(list(value.shape))
                print(f"{key:<40} | {shape_str:<20}")
                total_params += value.numel()
        
        print("-" * 60)
        print(f"Total Parameters: {total_params:,}")
        
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    inspect_pth("checkpoints/resnet_mnist.pth")
