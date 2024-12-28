import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np


def calculate_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std

if __name__ == '__main__':
    # 加载数据集
    ROOT_TRAIN = 'train'
    # 将图像的像素值归一化到[-1,1]之间
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # Compose()：将多个transforms的操作整合在一起
    train_transform = transforms.Compose([
        # Resize()：把给定的图像随机裁剪到指定尺寸
        transforms.Resize((224, 224)),
        # RandomVerticalFlip()：以0.5的概率竖直翻转给定的PIL图像
        # transforms.RandomVerticalFlip(),
        # ToTensor()：数据转化为Tensor格式
        transforms.ToTensor(),
        normalize])
    dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
    mean, std = calculate_mean_std(dataset)
    # 保存数据集均值和标准差
    np.save('mean.npy', mean.numpy())
    np.save('std.npy', std.numpy())
