import os
import re
import time
import requests
from numpy.random import random
import pandas as pd


def get_image_from_baidu(keyword, page_num, save_path):
    """
    从百度图片中爬取图片
    :param keyword: 搜索关键字
    :param page_num: 爬取页数
    :param save_path: 保存路径
    :return: None
    """
    image_num = 0
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'Host': 'image.baidu.com',
        'Referer': 'https://image.baidu.com/',
        'Cookie': 'BDqhfp=%E7%94%98%E8%8D%89%20%E4%B8%AD%E8%8D%AF%26%260-10-1undefined%26%260%26%261; BIDUPSID=5C7C29D789FF168EE73219CE1C6BBFC6; PSTM=1693559486; BAIDUID=5C7C29D789FF168EE73219CE1C6BBFC6:SL=0:NR=10:FG=1; BDUSS=3FWT2RURFNBaW1jdDJrR1BUREFyNU9VNWFuUXpaRHJQaXQ5Q1BReGFoN0h6NGRsSVFBQUFBJCQAAAAAAAAAAAEAAAA2ZDxTyOXRxbXEwffPyQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMdCYGXHQmBlS; BDUSS_BFESS=3FWT2RURFNBaW1jdDJrR1BUREFyNU9VNWFuUXpaRHJQaXQ5Q1BReGFoN0h6NGRsSVFBQUFBJCQAAAAAAAAAAAEAAAA2ZDxTyOXRxbXEwffPyQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMdCYGXHQmBlS; MCITY=-194%3A; H_PS_PSSID=39712_39817_39843_39903_39909_39934_39936_39933_39945_39940_39939_39930_39781_39998; H_WISE_SIDS=39712_39817_39843_39903_39909_39934_39936_39933_39945_39940_39939_39930_39781_39998; H_WISE_SIDS_BFESS=39712_39817_39843_39903_39909_39934_39936_39933_39945_39940_39939_39930_39781_39998; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; firstShowTip=1; cleanHistoryStatus=0; BAIDUID_BFESS=5C7C29D789FF168EE73219CE1C6BBFC6:SL=0:NR=10:FG=1; PSINO=7; delPer=1; BA_HECTOR=ag8l2ga0ag8ha505848hag8g1io2dqc1r; ZFY=4Y6:BFuBG4:A2Ns30MzNT8DRtbf5yajieUHmjX8RpKZ10:C; BDRCVFR[X_XKQks0S63]=mk3SLVN4HKm; BDRCVFR[dG2JNJb_ajR]=mk3SLVN4HKm; BDRCVFR[-pGxjrCMryR]=mk3SLVN4HKm; indexPageSugList=%5B%22%E7%94%98%E8%8D%89%20%E4%B8%AD%E8%8D%AF%22%2C%22%E7%99%BD%E5%9E%A9%20%E4%B8%AD%E8%8D%AF%22%2C%22%E7%81%B5%E8%8A%9D%22%2C%22%E9%95%BF%E7%94%9F%20%E4%B8%AD%E8%8D%AF%22%2C%22%E9%95%BF%E7%94%9F%22%2C%22%E4%B8%AD%E8%8D%AF%22%2C%22%E5%85%9A%E5%8F%82%22%2C%22%E6%9E%B8%E6%9D%9E%22%2C%22%E7%99%BD%E6%9C%AF%22%5D; userFrom=null; ab_sr=1.0.1_ZmMzY2Y1M2E4ZGI4ZTA4OTkyZDcyYzc5ZTE3YWI3ZjQwZGIxZjIxZjNmZTYwMTkzMDI0MGI5MDIzYzgwMmUxMGMwODE3NmI1ZTI1MDczNDBkMGEwOTNhODRmZWQzNDFhNjA4MDE0NTI5MWU4ZmFiZmM0YThkZGFiMjI5ODJlNzg3N2Q1MjJmMjRjNTg0ZjE0Yjc3OGE3NjYyZGYwZWIwMQ=='
    }

    url = 'https://image.baidu.com/search/acjson?'

    for i in range(page_num):
        time.sleep(random() * 5)
        param = {
            'tn': 'resultjson_com',
            'logid': '8330560121795637414',  # 日志id
            'ipn': 'rj',
            'ct': '201326592',
            'is': '',
            'fp': 'result',
            'fr': '',
            'word': keyword + ' 中药',
            'queryWord': keyword + ' 中药',
            'cl': 2,
            'lm': -1,
            'ie': 'utf-8',
            'oe': 'utf-8',
            'adpicid': '',
            'st': -1,
            'z': '',
            'ic': 0,
            'hd': '',
            'latest': '',
            'copyright': '',
            's': '',
            'se': '',
            'width': '',
            'height': '',
            'face': 0,
            'istype': 2,
            'qc': '',
            'nc': 1,
            'expermode': '',
            'nojc': '',
            'isAsync': '',
            'pn': i * 60,
            'rn': 30,  # 每页显示的图片数量
            'gsm': '3c',
            '1702968572427': ''
        }
        # 获取网页源代码
        request = requests.get(url, headers=headers, params=param)
        request.encoding = 'utf-8'
        html = request.text
        # 匹配thumburl=“”中的图片链接
        image_url_list = re.findall('"thumburl="(.*?)",', html, re.S)
        # 判断是否已存在，创建文件夹
        if not os.path.exists(save_path + '/' + keyword):
            os.makedirs(save_path + '/' + keyword)
        # 保存图片
        for image_url in image_url_list:
            image_data = requests.get(image_url).content
            with open(save_path + '/' + keyword + '/' + f'{image_num:06d}.jpg', 'wb') as f:
                f.write(image_data)
                print(f'正在下载{keyword}第' + str(image_num) + '张图片')
            image_num += 1


if __name__ == '__main__':
    # 读取文件，获取中药名
    df = pd.read_csv('../data/traditional_chinese_medicine_data.csv')
    # 爬取各中药的图片
    for name in df.name:
        keyword = name
        page_num = 20
        save_path = '../images/original'
        get_image_from_baidu(keyword, page_num, save_path)
