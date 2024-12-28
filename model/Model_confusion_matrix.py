import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn
from torchvision import models
from utils import load_data
import numpy as np
import torch.nn.functional as F
import seaborn as sns


# 定义函数生成混淆矩阵
def generate_confusion_matrix(dataloader, model, class_names):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(labels.cpu().tolist())

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 计算每个类别的样本数量
    class_counts = np.sum(cm, axis=1)

    # 计算每个格子的百分比
    cm_percent = cm / class_counts[:, np.newaxis]

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent , annot=True, fmt='.2f', cmap="Blues", cbar=True)
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names, rotation=45)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix Heatmap with Probabilities')
    # 保存混淆矩阵
    plt.savefig(f'trained_model/{type(model).__name__}/confusion_matrix.png', dpi=1500, format='png')
    plt.show()


if __name__ == '__main__':
    # 解决中文显示问题
    # 运行配置参数中的字体（font）为黑体（SimHei）
    plt.rcParams['font.sans-serif'] = ['simHei']
    # 运行配置参数总的轴（axes）正常显示正负号（minus）
    plt.rcParams['axes.unicode_minus'] = False

    # 加载数据
    train_dataloader, val_dataloader = load_data()

    # 如果有NVIDA显卡，可以转到GPU训练，否则用CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    AlexNet_path = 'trained_model/AlexNet/best_model.pth'
    AlexNet = models.alexnet(pretrained=False)
    AlexNet.classifier[6] = nn.Linear(AlexNet.classifier[6].in_features, 11)
    AlexNet.load_state_dict(torch.load(AlexNet_path, map_location=torch.device('cpu')))  # 加载模型权重
    AlexNet.to(device)

    ResNet_path = 'trained_model/ResNet/best_model.pth'
    ResNet = models.resnet50(pretrained=False)
    ResNet.fc = nn.Linear(ResNet.fc.in_features, 11)
    ResNet.load_state_dict(torch.load(ResNet_path, map_location=torch.device('cpu')))  # 加载模型权重
    ResNet.to(device)

    MobileNetV3_path = 'trained_model/MobileNetV3/best_model.pth'
    MobileNetV3 = models.mobilenet_v3_large(pretrained=False)
    MobileNetV3.classifier[3] = nn.Linear(MobileNetV3.classifier[3].in_features, 11)
    MobileNetV3.load_state_dict(torch.load(MobileNetV3_path, map_location=torch.device('cpu')))  # 加载模型权重
    MobileNetV3.to(device)

    # 生成混淆矩阵
    generate_confusion_matrix(val_dataloader, AlexNet, val_dataloader.dataset.classes)
    generate_confusion_matrix(val_dataloader, ResNet, val_dataloader.dataset.classes)
    generate_confusion_matrix(val_dataloader, MobileNetV3, val_dataloader.dataset.classes)
