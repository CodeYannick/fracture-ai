import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import get_resnet_mnist_model
import os
import ssl

# 修复 SSL 问题
ssl._create_default_https_context = ssl._create_unverified_context

def train_finetune():
    # 1. 配置
    # 假设我们要识别：10个数字 + 26个大写字母 + 26个小写字母 = 62类
    # EMNIST ByClass 数据集就是这种结构
    NEW_NUM_CLASSES = 62 
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0005 # 微调通常使用更小的学习率
    EPOCHS = 5
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 准备数据
    # 这里以 EMNIST 为例，它包含了数字和字母
    # 注意：EMNIST 的 split='byclass' 有 62 类
    print("Loading EMNIST dataset (ByClass split)...")
    
    # EMNIST 数据集默认是旋转了90度并翻转的，需要修正
    # 另外添加数据增强以提高泛化能力
    train_transform = transforms.Compose([
        transforms.RandomRotation(15), # 随机旋转 +/- 15度
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # 随机平移和缩放
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x.transpose(-2, -1) # 修正 EMNIST 旋转问题 (C, H, W) -> (C, W, H)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x.transpose(-2, -1) # 修正 EMNIST 旋转问题
    ])
    
    # 注意: torchvision 的 EMNIST url 有时不稳定，如果下载失败，可能需要手动下载
    # 这里只是示例，实际运行时如果没有 EMNIST 数据会报错
    try:
        train_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=train_transform)
        test_dataset = datasets.EMNIST(root='./data', split='byclass', train=False, download=True, transform=test_transform)
    except Exception as e:
        print(f"无法自动下载 EMNIST: {e}")
        print("您可以按照 README 说明使用 ImageFolder 加载自己的字母+数字数据集。")
        return

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 加载旧模型并进行修改
    print("Loading pre-trained MNIST model...")
    
    # (A) 先初始化一个 10 类的模型 (旧结构)
    model = get_resnet_mnist_model(num_classes=10)
    
    # (B) 加载之前训练好的权重
    checkpoint_path = "checkpoints/resnet_mnist.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Pre-trained weights loaded.")
    else:
        print("Warning: Pre-trained weights not found! Training from scratch.")

    # (C) 修改全连接层以适配新的类别数 (62)
    # 这一步会丢弃旧的 fc 层权重，并随机初始化新的 62 类 fc 层
    # 前面的卷积层权重保留，这就叫 "迁移学习"
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NEW_NUM_CLASSES)
    
    model = model.to(device)

    # 4. 定义优化器
    
    # 计算类别权重以解决样本不平衡问题
    # 简单的策略：给数字类别 (0-9) 更高的权重，因为用户希望能准确识别数字
    # 默认权重为 1.0，我们将数字的权重设为 2.5
    class_weights = torch.ones(NEW_NUM_CLASSES).to(device)
    class_weights[0:10] = 2.5 
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 方案 A: 训练所有层 (Fine-tuning) -> 也就是这里用的
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 方案 B: 冻结卷积层，只训练 fc 层 (Feature Extraction)
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.fc.weight.requires_grad = True
    # model.fc.bias.requires_grad = True
    # optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # 学习率调度器：每 2 个 epoch 衰减一次学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # 5. 训练循环 (同 train.py)
    print(f"Starting fine-tuning for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 这里的 transpose 已经移到了 transform 中处理
            # images = images.transpose(2, 3) 

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 更新学习率
        scheduler.step()
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1} Accuracy: {epoch_acc:.2f}%")

    # 6. 保存新模型
    save_path = "checkpoints/resnet_emnist_62class.pth"
    torch.save(model.state_dict(), save_path)
    print(f"New model saved to {save_path}")

if __name__ == "__main__":
    train_finetune()
