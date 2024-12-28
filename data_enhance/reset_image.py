import os
import cv2
import numpy as np
import pandas as pd


def load_image(name, load_path):
    """
    加载图片数据至列表
    :param name: 图片名称
    :param load_path: 加载图片的路径
    :return: 图片列表
    """
    file_name_list = os.listdir(f'{load_path}/{name}')
    image_list = []
    for file_name in file_name_list:
        image_list.append(cv2.imdecode(
            np.fromfile(
                f'{load_path}/{name}/{file_name}',
                dtype=np.uint8), 1))
    return image_list


def write_image(image_list, save_path, name):
    """
    将图片列表写入文件夹
    :param image_list: 图片列表
    :param save_path: 保存图片的路径
    :param name: 图片类别名称
    :return: None
    """
    for i, n in enumerate(image_list):
        cv2.imencode('.jpg', n)[1].tofile(
            f'{save_path}/{name}/{i:06d}.jpg')


def reshape_image(image):
    """
    将图片转换为256*256的大小
    :param image: 图片
    :return: 转换后的图片
    """
    return cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)


def compare_images(image1, image2):
    """
    将两张图片转为同样的大小后计算相似度
    :param image1: 图片1
    :param image2: 图片2
    :return: 相似度
    """
    # 计算彩色图像的直方图
    image1, image2 = reshape_image(image1), reshape_image(image2)
    hist1 = cv2.calcHist([image1], [2], None, [255], [0, 255])
    hist2 = cv2.calcHist([image2], [2], None, [255], [0, 255])
    # 计算直方图的相似度
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity


def remove_repeat_images(image_list, threshold=0.9):
    """
    去除重复的图片并调整图片为相同的大小
    :param image_list: 图片列表
    :param threshold: 相似度阈值，默认为0.9
    :return: 去重后的图片列表
    """
    # 创建一个新的列表来存储不重复的图片
    num = 0
    broken_image_num = 0
    unique_images = [reshape_image(image_list[0])]
    for image in image_list[1:]:
        # 检查图片是否与列表中的其他图片相似
        try:
            if not any(
                    compare_images(reshape_image(image), unique_image) > threshold for unique_image in unique_images):
                # 如果没有找到相似的图片，我们就将这张图片添加到列表中
                unique_images.append(reshape_image(image))
                num += 1
                if num % 100 == 0:
                    print(f'添加{num}张图片')
        except Exception as e:
            broken_image_num += 1
            print(f"有{broken_image_num}张图片损坏")
    return unique_images


if __name__ == '__main__':
    # 读取数据
    data = pd.read_csv('../data/traditional_chinese_medicine_data.csv')
    load_path = '../images/original'
    save_path = '../images/processing'
    for name in data.name:
        # 创建文件夹
        if not os.path.exists(f'{name}'):
            os.makedirs(f'{name}')
        # 读取图片
        image_list = load_image(name, load_path)
        # 处理图片
        new_images_list = remove_repeat_images(image_list)
        # 保存图片
        write_image(new_images_list, save_path, name)
