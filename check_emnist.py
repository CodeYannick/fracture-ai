import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import ssl

# Fix SSL
ssl._create_default_https_context = ssl._create_unverified_context

def check_emnist():
    print("Downloading EMNIST to check orientation...")
    try:
        data = datasets.EMNIST(root='./data', split='byclass', train=True, download=True)
    except Exception as e:
        print(f"Error loading EMNIST: {e}")
        return

    print("Saving first 5 images...")
    for i in range(5):
        img, label = data[i]
        # EMNIST images are PIL images
        img.save(f"emnist_sample_{i}.png")
        print(f"Saved emnist_sample_{i}.png (Label: {label})")

if __name__ == "__main__":
    check_emnist()
