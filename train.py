import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from model import SimpleCNN
from dataset import CustomImageDataset
from improved_model import ImprovedCNN


def train_model(num_epochs, learning_rate, batch_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("Using CPU")

    # 数据增强和预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据集
    train_dataset = CustomImageDataset(root_dir='data/train', transform=transform)
    val_dataset = CustomImageDataset(root_dir='data/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = ImprovedCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        print(f'Validation Accuracy: {100 * correct / total}%')

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')