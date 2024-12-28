import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def load_data():
    """
    加载数据集
    :return: train_dataloader, val_dataloader
    """
    ROOT_TRAIN = '../data/train'
    ROOT_VAL = '../data/test'

    # 将图像的像素值归一化到[-1,1]之间
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # Compose()：将多个transforms的操作整合在一起
    train_transform = transforms.Compose([
        # 把给定的图像裁剪到指定尺寸
        transforms.Resize((224, 224)),
        # 随机翻转，图像以0.5的概率竖直翻转给定的PIL图像
        # transforms.RandomVerticalFlip(),
        # 随机修改图像的亮度、对比度和饱和度
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        # 随机旋转图像
        transforms.RandomRotation(10),
        # 随机裁剪图像
        # transforms.RandomResizedCrop(224),
        # ToTensor()：数据转化为Tensor格式
        transforms.ToTensor(),
        # 将图像的像素值归一化到[-1,1]之间
        normalize])

    val_transform = transforms.Compose([
        # Resize()：把给定的图像随机裁剪到指定尺寸
        transforms.Resize((224, 224)),
        # ToTensor()：数据转化为Tensor格式
        transforms.ToTensor(),
        # 将图像的像素值归一化到[-1,1]之间
        normalize])

    # 加载训练数据集
    # ImageFolder：假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名，其构造函数如下：
    # ImageFolder(root, transform=None, target_transform=None, loader=default_loader)
    # root：在root指定的路径下寻找图像，transform：对输入的图像进行的转换操作
    train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
    # DataLoader：将读取的数据按照batch size大小封装给训练集
    # dataset (Dataset)：加载数据的数据集
    # batch_size (int, optional)：每个batch加载多少个样本(默认: 1)
    # shuffle (bool, optional)：设置为True时会在每个epoch重新打乱数据(默认: False)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 加载训练数据集
    val_dataset = ImageFolder(ROOT_VAL, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    return train_dataloader, val_dataloader


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    """
    训练函数
    :param dataloader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :return: train_loss, train_acc
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train().to(device)
    loss, current, n = 0.0, 0.0, 0
    # dataloader: 传入数据（数据包括：训练数据和标签）
    # enumerate()：用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中
    # enumerate返回值有两个：一个是序号，一个是数据（包含训练数据和标签）
    # x：训练数据（inputs）(tensor类型的），y：标签（labels）(tensor类型）
    for batch, (x, y) in enumerate(dataloader):
        # 前向传播
        image, y = x.to(device), y.to(device)
        # 计算训练值
        output = model(image)
        # 计算观测值（label）与训练值的损失函数
        cur_loss = loss_fn(output, y)
        # torch.max(input, dim)函数
        # input是具体的tensor，dim是max函数索引的维度，0是每列的最大值，1是每行的最大值输出
        # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
        _, pred = torch.max(output, axis=1)
        # 计算每批次的准确率
        # output.shape[0]为该批次的多少，output的一维长度
        # torch.sum()对输入的tensor数据的某一维度求和
        cur_acc = torch.sum(y == pred) / output.shape[0]

        # 反向传播
        # 清空过往梯度
        optimizer.zero_grad()
        # 反向传播，计算当前梯度
        cur_loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()
        # item()：得到元素张量的元素值
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    train_loss = loss / n
    train_acc = current / n
    # 计算训练的错误率
    print(f'train_loss:{train_loss}')
    # 计算训练的准确率
    print(f'train_acc:{train_acc}')
    return train_loss, train_acc


# 定义验证函数
def val(dataloader, model, loss_fn):
    """
    验证函数
    :param dataloader:
    :param model:
    :param loss_fn:
    :return: val_loss, val_acc
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss, current, n = 0.0, 0.0, 0
    model.eval()

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            # 前向传播
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

    val_loss = loss / n
    val_acc = current / n
    # 计算验证的错误率
    print(f'val_loss:{val_loss}')
    # 计算验证的准确率
    print(f'val_acc:{val_acc}')
    return val_loss, val_acc


# 定义画图函数
# 错误率
def matplot_loss(train_loss, val_loss):
    """
    损失函数图
    :param train_loss:
    :param val_loss:
    :return:
    """
    # 参数label = ''传入字符串类型的值，也就是图例的名称
    plt.plot(range(len(train_loss)), train_loss, label='train_loss')
    plt.plot(range(len(val_loss)), val_loss, label='val_loss')
    # loc代表了图例在整个坐标轴平面中的位置（一般选取'best'这个参数值）
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('训练集和验证集的loss值对比图')
    plt.show()


# 准确率
def matplot_acc(train_acc, val_acc):
    """
    正确率图
    :param train_acc:
    :param val_acc:
    :return:
    """
    plt.plot(range(len(train_acc)), train_acc, label='train_acc')
    plt.plot(range(len(val_acc)), val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('训练集和验证集的acc值对比图')
    plt.show()
