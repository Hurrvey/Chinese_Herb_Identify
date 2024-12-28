from torchsummary import summary
import torch
import torch.nn as nn
import torchvision.models as models
import sys


def model_summary(model, input_shape):
    # 保存原始的标准输出
    original_stdout = sys.stdout
    # 恢复原始的标准输出
    sys.stdout = original_stdout
    name = type(model).__name__
    # 重定向标准输出到一个文件
    with open(f'trained_model/{name}/{name}_summary.txt', 'w') as f:
        sys.stdout = f
        summary(model, (3, input_shape[0], input_shape[1]))


input_shape = [224, 224]  # 设置输入大小
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择是否使用GPU

AlexNet = models.alexnet(pretrained=True)
AlexNet.classifier[6] = nn.Linear(AlexNet.classifier[6].in_features, 11)
AlexNet.to(device)

MobileNetV3 = models.mobilenet_v3_large(pretrained=True)
MobileNetV3.classifier[3] = nn.Linear(MobileNetV3.classifier[3].in_features, 11)
MobileNetV3.to(device)

ResNet = models.resnet50(pretrained=True)
ResNet.fc = nn.Linear(ResNet.fc.in_features, 11)
ResNet.to(device)

model_summary(MobileNetV3, input_shape)
model_summary(AlexNet, input_shape)
model_summary(ResNet, input_shape)



