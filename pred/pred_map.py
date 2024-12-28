import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


def predict_and_plot_images(model, root_dir='../data/test'):
    # 获取所有子目录
    sub_dirs = [os.path.join(root_dir, o) for o in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, o))]

    # 创建一个新的画布
    fig = plt.figure(figsize=(12, 9))
    # 定义转换
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    # 定义标签
    tag = {0: '人参', 1: '列当', 2: '桔梗', 3: '沙参', 4: '玉竹', 5: '甘草', 6: '知母', 7: '肉苁蓉', 8: '锁阳',
           9: '黄精', 10: '黄耆'}

    # 遍历所有子目录
    for i, sub_dir in enumerate(sub_dirs):
        # 获取子目录中的所有图像文件
        image_files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, f))]

        # 随机选择一个图像文件进行预测
        image_file = random.choice(image_files)
        image = Image.open(image_file)
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_label = tag[predicted.item()]

        # 在画布上绘制图像
        ax = fig.add_subplot(3, 4, i+1)  # 修改这里，将画布分为3行4列
        img = mpimg.imread(image_file)
        ax.imshow(img)
        ax.axis('off')

        # 设置坐标轴的间隔
        plt.xticks(range(0, img.shape[1], 60))  # 设置x轴的间隔为60
        plt.yticks(range(0, img.shape[0], 60))  # 设置y轴的间隔为60

        # 在图像上添加真实的分类和预测分类
        ax.set_title(f'Real: {os.path.basename(sub_dir)}, Predicted: {predicted_label}')

    # 调整子图间隔
    plt.subplots_adjust(wspace=0.5, hspace=0.0001, top=0.9, bottom=0.1)

    # 保存画布到本地文件
    plt.savefig('predicted_images.png', dpi=1500, format='png')


if __name__ == '__main__':
    # 解决中文显示问题
    # 运行配置参数中的字体（font）为黑体（SimHei）
    plt.rcParams['font.sans-serif'] = ['simHei']
    # 运行配置参数总的轴（axes）正常显示正负号（minus）
    plt.rcParams['axes.unicode_minus'] = False
    # 调用函数
    ResNet_path = '../model/trained_model/ResNet/best_model.pth'
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 11)
    model.load_state_dict(torch.load(ResNet_path, map_location=torch.device('cpu')))  # 加载模型权重
    model.eval()
    predict_and_plot_images(model)
