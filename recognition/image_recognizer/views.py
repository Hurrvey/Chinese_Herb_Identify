from io import BytesIO
from django.shortcuts import render, redirect
from forms import ImageUploadForm
import torch
from PIL import Image
from torchvision import transforms
import re
import sys
import torch.nn as nn
import torchvision.models as models
import base64

sys.path.append(
    'D:/PycharmProjects/pythonProject/Traditional Chinese Medicine Identification/recognition/image_recognizer/')
from image_recognizer.models import Data


def format_description(description):
    """
    格式化文本信息
    :param description: 文本信息
    :return: 格式化后的文本信息
    """
    sections = re.findall(r'【(.*?)】(.*?)(?=【|$)', description, re.S)
    formatted_description = {}
    for title, content in sections:
        formatted_description.update({title: content})
    return formatted_description


def handle_uploaded_file(f):
    """
    保存上传的文件
    :param f: 上传的文件
    :return: 保存的文件名
    """
    with open('some_file_name.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return destination.name


def predict_image(image_path):
    """
    预测图像
    :param image_path: 图像路径
    :return: 预测结果
    """
    # 定义转换
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    # # 如果有NVIDA显卡，可以转到GPU训练，否则用CPU
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载模型
    ResNet_path = 'D:/PycharmProjects/pythonProject/Traditional Chinese Medicine Identification/model/trained_model/ResNet/best_model.pth'
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 11)
    model.load_state_dict(torch.load(ResNet_path, map_location=torch.device('cpu')))  # 加载模型权重
    # model.to(device)
    model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        tag = {0: '人参', 1: '列当', 2: '桔梗', 3: '沙参', 4: '玉竹', 5: '甘草', 6: '知母', 7: '肉苁蓉', 8: '锁阳',
               9: '黄精', 10: '黄耆'}
        return tag[predicted.item()]


def upload_image(request):
    """
    上传图像
    :param request: 请求
    :return: 上传图像页面
    """
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file_path = handle_uploaded_file(request.FILES['image'])
            result = predict_image(file_path)
            if result is not None:
                return redirect('result', result=result)
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})


def result(request, result):
    """
    结果页面
    :param request: 请求
    :param result: 预测结果
    :return: 结果页面
    """
    image = Data.objects.get(name=result)
    formatted_description = format_description(image.description)

    # 读取图像文件,并调整大小
    image_path = './some_file_name.jpg'
    image = Image.open(image_path)
    image = image.resize((224, 224))
    # 将图像转换为字节流
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    # 将字节流转换为base64编码
    encoded_image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render(request, 'result.html', {'result': result, 'description': formatted_description,
                                           'encoded_image_data': encoded_image_data})
