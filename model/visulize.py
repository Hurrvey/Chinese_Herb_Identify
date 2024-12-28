from torchviz import make_dot
import torch
import torch.nn as nn
import torchvision.models as models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择是否使用GPU
input_tensor = torch.randn(1, 3, 224, 224).to(device)  # 假设输入的是224*224的图片

AlexNet = models.alexnet(pretrained=False)
AlexNet.classifier[6] = nn.Linear(AlexNet.classifier[6].in_features, 11)
AlexNet.to(device)

MobileNet_v3 = models.mobilenet_v3_large(pretrained=False)
MobileNet_v3.classifier[3] = nn.Linear(MobileNet_v3.classifier[3].in_features, 11)
MobileNet_v3.to(device)

ResNet = models.resnet50(pretrained=False)
ResNet = nn.Linear(ResNet.fc.in_features, 11)
ResNet.to(device)

output1 = AlexNet(input_tensor)
dot1 = make_dot(output1)
dot1.view(filename='trained_model/AlexNet/AlexNet')

output2 = MobileNet_v3(input_tensor)
dot2 = make_dot(output2)
dot2.view(filename='trained_model/MobileNetV3/MobileNetV3')

output3 = MobileNet_v3(input_tensor)
dot3 = make_dot(output3)
dot3.view(filename='trained_model/ResNet/ResNet')
