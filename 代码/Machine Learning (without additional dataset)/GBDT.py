import os
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import json
import torch

# 加载分割索引文件
def load_split_indices(dataset, split_path="data_splits.json"):
    with open(split_path, "r") as f:
        split_data = json.load(f)

    # 从 split_data 中提取训练集、验证集和测试集索引
    train_data = Subset(dataset, split_data["train_indices"])
    val_data = Subset(dataset, split_data["val_indices"])
    test_data = Subset(dataset, split_data["test_indices"])
    print(f"数据分割索引已从 {split_path} 加载")
    return train_data, val_data, test_data

# 保存分割索引文件
def save_split_indices(split_data, split_path="data_splits.json"):
    with open(split_path, "w") as f:
        json.dump(split_data, f)
    print(f"数据分割索引已保存到 {split_path}")

# 划分数据集
def split_dataset(dataset, split_path="data_splits.json", train_ratio=0.90, val_ratio=0.05, test_ratio=0.05):
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_end = int(train_ratio * total_size)
    val_end = train_end + int(val_ratio * total_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    split_data = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices
    }

    save_split_indices(split_data, split_path)
    return train_indices, val_indices, test_indices

# 计算准确率
def calculate_accuracy(predictions, labels):
    return accuracy_score(labels, predictions)

# 绘制混淆矩阵
def plot_confusion_matrix(cm, labels, save_path="./image/xgboost/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.show()
    print(f"Confusion matrix saved to {save_path}")

# 绘制预测图像
def plot_images(images, labels, preds, ncols=5, save_path="./image/xgboost/25_images.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(ncols, ncols, figsize=(12, 12))
    axes = axes.flatten()
    for i in range(ncols * ncols):
        if i < len(images):
            axes[i].imshow(images[i].cpu().squeeze(), cmap="gray")
            axes[i].set_title(f"True: {labels[i]}, Pred: {preds[i]}")
            axes[i].axis("off")
        else:
            axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Prediction images saved to {save_path}")

# 主程序
if __name__ == '__main__':
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载训练集和测试集，并将它们合并
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([train_dataset, test_dataset])  # 合并训练集和测试集

    split_file = "data_splits.json"

    if os.path.exists(split_file):
        train_data, val_data, test_data = load_split_indices(full_dataset, split_file)
    else:
        train_indices, val_indices, test_indices = split_dataset(full_dataset, split_file)
        train_data = Subset(full_dataset, train_indices)
        val_data = Subset(full_dataset, val_indices)
        test_data = Subset(full_dataset, test_indices)

    # 数据加载器
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # 将训练数据和验证数据转换为适用于XGBoost的numpy数组
    def load_data_to_numpy(loader):
        images = []
        labels = []
        for img, lbl in loader:
            images.append(img.view(img.size(0), -1).cpu().numpy())  # 扁平化图像
            labels.append(lbl.cpu().numpy())
        return np.concatenate(images), np.concatenate(labels)


    # 加载训练集、验证集和测试集数据
    X_train, y_train = load_data_to_numpy(train_loader)
    X_val, y_val = load_data_to_numpy(val_loader)
    X_test, y_test = load_data_to_numpy(test_loader)

    # 合并训练集和验证集
    X_combined = np.concatenate([X_train, X_val], axis=0)  # 合并训练集和验证集
    y_combined = np.concatenate([y_train, y_val], axis=0)  # 合并对应标签

    # 更新后的训练数据
    X_train, y_train = X_combined, y_combined

    # 检测 GPU 并设置加速方式
    use_gpu = os.environ.get('CUDA_VISIBLE_DEVICES') is not None
    tree_method = 'gpu_hist' if use_gpu else 'hist'
    print(f"Using tree_method: {tree_method}")

    # 使用XGBoost进行训练
    clf = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        tree_method=tree_method,
        random_state=42
    )
    print("Training the XGBoost model...")
    clf.fit(X_train, y_train)

    # 预测
    print("Evaluating the model on test data...")
    y_pred = clf.predict(X_test)

    # 计算准确率
    accuracy = calculate_accuracy(y_pred, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    labels = [str(i) for i in range(10)]
    plot_confusion_matrix(cm, labels, save_path="./image/xgboost/confusion_matrix.png")

    # 随机挑选25张测试集图像并显示
    images, labels = next(iter(test_loader))
    outputs = clf.predict(images.view(images.size(0), -1).cpu().numpy())  # 扁平化图像并进行预测
    plot_images(images, labels, outputs, save_path="./image/xgboost/25_images.png")