import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from torchvision import datasets, transforms

# 从mmap文件中加载数据
def load_from_mmap(images_path, labels_path, image_shape):
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"mmap 文件未找到，请检查路径：\n{images_path}\n{labels_path}")
    images = np.memmap(images_path, dtype='float32', mode='r', shape=image_shape)
    labels = np.memmap(labels_path, dtype='int64', mode='r')
    return images, labels

# 计算准确率
def calculate_accuracy(predictions, labels):
    return accuracy_score(labels, predictions)

# 绘制混淆矩阵
def plot_confusion_matrix(cm, labels, save_path="./image/rf/large/confusion_matrix.png"):
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
def plot_images(images, labels, preds, ncols=5, save_path="./image/rf/large/25_images.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    axes = axes.flatten()
    for i in range(25):
        img = images[i].reshape(28, 28)  # 将数据恢复为28x28的图像
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"True: {labels[i]}, Pred: {preds[i]}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Prediction images saved to {save_path}")

# 主程序
if __name__ == '__main__':
    # mmap文件路径
    mmap_dir = "./mmap_data/large"
    train_images_path = os.path.join(mmap_dir, "rotated_train_images.dat")
    train_labels_path = os.path.join(mmap_dir, "rotated_train_labels.npy")

    # 获取样本数并明确图像形状
    train_labels = np.memmap(train_labels_path, dtype='int64', mode='r')
    num_train_samples = len(train_labels)  # 获取训练样本数
    train_image_shape = (num_train_samples, 1, 28, 28)  # 明确训练图像的形状

    # 从mmap文件加载训练集
    X_train, y_train = load_from_mmap(train_images_path, train_labels_path, train_image_shape)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # 将每个样本展平成一维

    # 加载测试集直接从PyTorch
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 将测试集数据转换为适配Random Forest的格式
    X_test = test_dataset.data.numpy().reshape(-1, 28 * 28).astype('float32')
    y_test = test_dataset.targets.numpy()

    # 使用随机森林进行训练
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Training the Random Forest model...")
    clf.fit(X_train_flat, y_train)

    # 预测
    print("Evaluating the model on test data...")
    y_pred = clf.predict(X_test)

    # 计算准确率
    accuracy = calculate_accuracy(y_pred, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    labels = [str(i) for i in range(10)]
    plot_confusion_matrix(cm, labels, save_path="./image/rf/large/confusion_matrix.png")

    # 随机挑选25张测试集图像并显示
    indices = np.random.choice(len(X_test), size=25, replace=False)
    images_sample = X_test[indices]
    labels_sample = y_test[indices]
    preds_sample = y_pred[indices]
    plot_images(images_sample, labels_sample, preds_sample, save_path="./image/rf/large/25_images.png")
