import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import get_resnet_mnist_model
import time
from tqdm import tqdm
import os
import ssl

# 修复 Mac 上的 SSL 证书验证失败错误 (用于下载数据集)
ssl._create_default_https_context = ssl._create_unverified_context

def train():
    # 设置设备 - 如果可用，优先使用 Mac 的 MPS (Metal Performance Shaders) 加速
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) acceleration (使用 MPS Metal 加速)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration (使用 CUDA 加速)")
    else:
        device = torch.device("cpu")
        print("Using CPU (使用 CPU)")

    # 超参数 (Hyperparameters)
    BATCH_SIZE = 128      # 批大小
    LEARNING_RATE = 0.001 # 学习率
    EPOCHS = 5            # 训练轮数

    # 数据预处理 (Data Transformations)
    transform = transforms.Compose([
        transforms.ToTensor(), # 转为 Tensor
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 数据集的均值和标准差归一化
    ])

    # 加载 MNIST 数据集
    print("Loading MNIST dataset... (正在加载 MNIST 数据集...)")
    
    # --- 如果您想使用自己的数据集 (Custom Dataset) ---
    # 方法 1: ImageFolder (推荐，如果您的数据是按文件夹分类的)
    # 目录结构:
    # data/my_dataset/train/class_0/xxx.jpg
    # data/my_dataset/train/class_1/xxx.jpg
    # 代码:
    # train_dataset = datasets.ImageFolder(root='./data/my_dataset/train', transform=transform)
    # test_dataset = datasets.ImageFolder(root='./data/my_dataset/val', transform=transform)

    # 方法 2: 自定义 Dataset 类 (如果您有图片和对应的标签文件 txt/csv)
    # from torch.utils.data import Dataset
    # from PIL import Image
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, txt_file, root_dir, transform=None):
            self.img_labels = [] # 读取 txt_file 到列表
            self.root_dir = root_dir
            self.transform = transform
        def __len__(self):
            return len(self.img_labels)
        def __getitem__(self, idx):
            img_path = os.path.join(self.root_dir, self.img_labels[idx][0])
            image = Image.open(img_path).convert('L') # 转为灰度
            label = int(self.img_labels[idx][1])
            if self.transform:
                image = self.transform(image)
            return image, label
    
    # 默认使用 MNIST
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = get_resnet_mnist_model().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss() # 交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam 优化器

    # 训练循环 (Training Loop)
    print(f"Starting training for {EPOCHS} epochs... (开始训练 {EPOCHS} 轮...)")
    
    for epoch in range(EPOCHS):
        model.train() # 设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # 前向传播 (Forward pass)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化 (Backward pass and optimize)
            optimizer.zero_grad() # 清空梯度
            loss.backward()       # 计算梯度
            optimizer.step()      # 更新参数

            # 统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # 验证阶段 (Validation Phase)
        model.eval() # 设置为评估模式
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad(): # 不计算梯度
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(test_loader)
        
        end_time = time.time()
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.2f}% | "
              f"Time: {end_time - start_time:.2f}s")

    # 保存模型
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/resnet_mnist.pth")
    print("Model saved to checkpoints/resnet_mnist.pth (模型已保存)")

if __name__ == "__main__":
    train()
