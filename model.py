import torch
import torch.nn as nn
from torchvision import models

def get_resnet_mnist_model():
    # Load ResNet18
    # We don't need pretrained weights for such a simple dataset like MNIST, 
    # but using them can speed up convergence. 
    # However, since we are changing the first layer heavily, training from scratch is also fine.
    # Let's use no weights for a "clean" experiment or weights=None.
    model = models.resnet18(weights=None)
    
    # Modify the first convolution layer
    # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # MNIST images are 28x28 grayscale (1 channel).
    # To avoid aggressive downsampling on small images, we can change kernel size and stride.
    # Standard ResNet: 7x7 conv stride 2 -> 14x14 output.
    # We can change it to 3x3 conv stride 1 to keep spatial dimensions larger longer if we wanted,
    # but for a simple "use ResNet" demo, just changing channels is the minimal change.
    # However, 28x28 -> (stride 2) 14x14 -> (maxpool stride 2) 7x7 ... eventually it gets very small (1x1) quickly.
    # ResNet18 structure:
    # Conv1 (s2) -> 14x14
    # MaxPool (s2) -> 7x7
    # Layer1 -> 7x7
    # Layer2 (s2) -> 4x4
    # Layer3 (s2) -> 2x2
    # Layer4 (s2) -> 1x1
    # AvgPool -> 1x1
    # This actually works out exactly to 1x1 feature map before the final FC.
    
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify the final fully connected layer
    # Original: nn.Linear(512, 1000)
    # MNIST has 10 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    return model
