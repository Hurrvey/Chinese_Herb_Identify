import torch
from torch import nn
from torchvision import models
from utils import load_data
from torch.autograd import Variable
from sklearn.metrics import classification_report


def evaluate_model(model, dataloader):
    # 如果有NVIDA显卡，可以转到GPU训练，否则用CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()  # 设置为评估模式
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred


if __name__ == '__main__':
    # 加载数据
    train_dataloader, val_dataloader = load_data()

    # 如果有NVIDA显卡，可以转到GPU训练，否则用CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
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

    AlexNet_path = 'trained_model/AlexNet/best_model.pth'
    AlexNet = models.alexnet(pretrained=False)
    AlexNet.classifier[6] = nn.Linear(AlexNet.classifier[6].in_features, 11)
    AlexNet.load_state_dict(torch.load(AlexNet_path, map_location=torch.device('cpu')))  # 加载模型权重
    AlexNet.to(device)

    # 计算AlexNet的评估指标
    y_true, y_pred = evaluate_model(AlexNet, val_dataloader)
    report = classification_report(y_true, y_pred, digits=4)
    with open('trained_model/AlexNet/AlexNet_report.txt', 'w') as f:
        f.write("AlexNet:\n")
        f.write(report)

    # 计算MobileNetV3的评估指标
    y_true, y_pred = evaluate_model(MobileNetV3, val_dataloader)
    report = classification_report(y_true, y_pred, digits=4)
    with open('trained_model/MobileNetV3/MobileNetV3_report.txt', 'w') as f:
        f.write("MobileNetV3:\n")
        f.write(report)

    # 计算ResNet50的评估指标
    y_true, y_pred = evaluate_model(ResNet, val_dataloader)
    report = classification_report(y_true, y_pred, digits=4)
    with open('trained_model/ResNet/ResNet_report.txt', 'w') as f:
        f.write("ResNet:\n")
        f.write(report)
