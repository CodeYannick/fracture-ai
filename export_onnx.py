import torch
from model import get_resnet_mnist_model
import os

def export_to_onnx():
    # Load the model structure
    model = get_resnet_mnist_model()
    
    # Load the trained weights
    checkpoint_path = "checkpoints/resnet_mnist.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print("Warning: Checkpoint not found, exporting with random weights.")
    
    model.eval()
    
    # Create a dummy input tensor matching the input shape (Batch, Channels, Height, Width)
    # MNIST is (1, 28, 28)
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Export
    output_path = "checkpoints/resnet_mnist.onnx"
    torch.onnx.export(model, 
                      dummy_input, 
                      output_path, 
                      input_names=['input'], 
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    
    print(f"Model exported to {output_path}")
    print("You can view this file using Netron (https://netron.app)")

if __name__ == "__main__":
    export_to_onnx()
