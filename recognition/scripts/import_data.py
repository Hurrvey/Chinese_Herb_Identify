import csv
import sys

sys.path.append(
    'D:/PycharmProjects/pythonProject/Traditional Chinese Medicine Identification/recognition/image_recognizer')
from image_recognizer.models import Data  # 替换为应用名


def run():
    with open(
            'D:/PycharmProjects/pythonProject/Traditional Chinese Medicine Identification/data/traditional_chinese_medicine_data.csv',
            'r', encoding='utf-8') as file:  # 替换为你的文件路径
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            _, created = Data.objects.get_or_create(
                name=row[0],
                description=row[1]
            )
