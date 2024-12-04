import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.metrics import confusion_matrix

# 设置设备，选择 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理：转换为1通道并规范化
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整为ResNet所需的输入尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 单通道的均值和标准差
])

# 保存分割索引到文件
def save_split_indices(train_indices, val_indices, test_indices, save_path="data_splits.json"):
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    split_data = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices
    }
    with open(save_path, "w") as f:
        json.dump(split_data, f)
    print(f"数据分割索引已保存到 {save_path}")

# 加载分割索引文件
def load_split_indices(dataset, split_path="data_splits.json"):
    with open(split_path, "r") as f:
        split_data = json.load(f)
    train_data = Subset(dataset, split_data["train_indices"])
    val_data = Subset(dataset, split_data["val_indices"])
    test_data = Subset(dataset, split_data["test_indices"])
    print(f"数据分割索引已从 {split_path} 加载")
    return train_data, val_data, test_data

# 数据分割并保存索引
def split_and_save_dataset(dataset, save_path="data_splits.json", train_ratio=0.9, val_ratio=0.05):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    save_split_indices(train_indices, val_indices, test_indices, save_path)
    return train_indices, val_indices, test_indices

# 计算混淆矩阵
def compute_confusion_matrix(model, loader):
    all_preds = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm

# 绘制混淆矩阵
def plot_confusion_matrix(cm, labels, save_path="./image/cnn/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.show()
    print(f"Confusion matrix saved to {save_path}")

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# 定义CNN模型，使用残差连接
class CNNWithResiduals(nn.Module):
    def __init__(self):
        super(CNNWithResiduals, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        # 这里的 fc 层输入大小会根据输入图片大小计算
        self.fc = nn.Linear(512 * 7 * 7, 10)  # 10 类输出

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, stride=1)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 计算卷积层的输出尺寸
        x = torch.flatten(x, 1)  # 将 x 展平为 [batch_size, features]
        x = self.fc(x)
        return x


# 计算模型在给定数据加载器上的准确率
def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 训练一个epoch
def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    return train_loss, train_accuracy

# 验证一个epoch
def validate_one_epoch(model, val_loader, criterion):
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    return val_loss, val_accuracy

# 绘制训练和验证过程的损失与准确率图
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir="./image/cnn"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics.png"))
    plt.show()

# 保存训练好的模型
def save_model(model, save_path="./model/cnn"):
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "cnn_model.pth"))
    print(f"模型已保存到 {save_path}")

# 显示图像并保存
def show_images(images, labels, preds, save_path, ncols=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    axes = axes.flatten()
    for i in range(25):
        axes[i].imshow(images[i].cpu().squeeze(), cmap="gray")
        axes[i].set_title(f"True: {labels[i]}, Pred: {preds[i]}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# 主训练流程
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies

# 主程序
if __name__ == '__main__':
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    split_file = "data_splits.json"

    if os.path.exists(split_file):
        train_data, val_data, test_data = load_split_indices(full_dataset, split_file)
    else:
        train_indices, val_indices, test_indices = split_and_save_dataset(full_dataset, save_path=split_file)
        train_data = Subset(full_dataset, train_indices)
        val_data = Subset(full_dataset, val_indices)
        test_data = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = CNNWithResiduals()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=10
    )

    test_accuracy = calculate_accuracy(test_loader, model)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 保存训练图像和模型
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir="./image/cnn")
    save_model(model, save_path="./model/cnn")

    cm = compute_confusion_matrix(model, test_loader)
    labels = [str(i) for i in range(10)]
    plot_confusion_matrix(cm, labels, save_path="./image/cnn/confusion_matrix.png")

    # 随机挑选25张验证集图像
    images, labels = next(iter(val_loader))
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)
    show_images(images, labels, preds, save_path="./image/cnn/25_images.png")
