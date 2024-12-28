import os
import cv2
import numpy as np
import pandas as pd
from reset_image import load_image, write_image


def flip_image(image):
    """
    翻转图片
    :param image: 图片
    :return: 翻转后的图片列表
    """
    flip_image_list = []
    for flip in [0, 1, -1]:
        flip_image = cv2.flip(image, flip)
        flip_image_list.append(flip_image)
    return flip_image_list


def rotate_image(image):
    """
    旋转图片, 旋转角度为90, 180, 270
    :param image: 图片
    :return: 图片列表, 包括旋转90, 180, 270度后的图片
    """
    rotate_image_list = []
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotate_image = cv2.warpAffine(image, M, (w, h))
        rotate_image_list.append(rotate_image)
    return rotate_image_list


def random_brightness(image):
    """
    随机变化图片的亮度
    :param image: 图片
    :return: 亮度变化后的图片
    """
    # 随机生成亮度调整因子
    # np.random.uniform(low, high)生成一个low到high之间的随机浮点数
    brightness_factor = np.random.uniform(-9, 9)
    if brightness_factor == 0:
        return random_brightness(image)
    # 创建一个与原图像形状相同的亮度调整矩阵
    brightness_matrix = np.ones(image.shape, dtype=np.uint8) * int(brightness_factor)

    # 对图像的每个像素值进行加法操作
    image_brightness = cv2.add(image, brightness_matrix.astype(np.uint8))

    # 对调整后的像素值进行裁剪，使其落在合法范围内[0,255]
    image_brightness = np.clip(image_brightness, 0, 255)

    return image_brightness


def random_hue(image):
    """
    随机变化图片的色度
    :param image: 图片
    :return: 色度变化后的图片
    """
    # 将图像从BGR颜色空间转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 随机生成色度调整因子
    hue_val = np.random.randint(-5, 5) + 1
    if hue_val == 0:
        return random_hue(image)
    # 对色度通道进行调整
    hsv[:, :, 0] = hsv[:, :, 0] + hue_val

    # 对调整后的色度值进行裁剪，使其落在合法范围内[0,179]
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)

    # 将图像从HSV颜色空间转换回BGR颜色空间
    image_hue = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image_hue


def random_contrast(image):
    """
    随机变化图片的对比度
    :param image: 图片
    :return: 对比度变化后的图片
    """
    # 随机生成对比度调整因子
    contrast_factor = np.random.uniform(0.5, 1.1)

    # 将图像转换为浮点数
    image = image.astype(float)

    # 对图像的每个像素值进行乘法操作
    image_contrast = image * contrast_factor

    # 对调整后的像素值进行裁剪，使其落在合法范围内[0,255]
    image_contrast = np.clip(image_contrast, 0, 255)

    # 将图像转换回uint8类型
    image_contrast = image_contrast.astype(np.uint8)

    return image_contrast


def random_image(image):
    """
    对图片进行亮度，色度，对比度等变化
    :param image: 图片
    :return: 随机变化后的图片
    """
    # 图片亮度
    image1 = random_brightness(image)

    # 色度变化
    image2 = random_hue(image)

    # 对比度变化
    image3 = random_contrast(image)

    return image3


def expand_image(image_list):
    """
    扩充数据集, 包括旋转, 翻转, 亮度, 对比度, 锐化
    :param image_list: 图片列表
    :return: 扩充后的图片列表
    """
    new_images_list = []
    for image in image_list:
        temple_list = []
        # 翻转
        temple_list.extend(flip_image(image))
        # 旋转
        temple_list.extend(rotate_image(image))
        # 随机生成3个0到6的数字
        for n in np.random.randint(0, 6, 3):
            # 亮度, 色度, 对比度等变化
            temple_list.append(random_image(temple_list[n]))
        new_images_list.extend(temple_list)
    return new_images_list


if __name__ == '__main__':
    data = pd.read_csv('../data/traditional_chinese_medicine_data.csv')
    load_path = '../images/processing'
    save_path = '../images/final'
    for name in data.name:
        # 创建文件夹
        if not os.path.exists(f'{save_path}/{name}'):
            os.makedirs(f'{save_path}/{name}')
        # 读取图片
        image_list = load_image(name, load_path)
        # 处理图片
        new_image_list = expand_image(image_list)
        # 保存图片
        write_image(new_image_list, save_path, name)
