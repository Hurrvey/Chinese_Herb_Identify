import time
import requests
from bs4 import BeautifulSoup
from numpy.random import random
import pandas as pd


def get_chinese_meidicine_text(Host, href):
    """
    获取中药文本
    :return: 中药文本
    """
    time.sleep(random() * 5)  # 随机休眠
    url = 'http://' + Host + href
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'Host': Host,
        'Connection': 'keep-alive'
    }
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    p_list = soup.select('.spider p')  # 选取class=spider下的所有p标签
    del p_list[0]
    chinese_medicine_text = ''
    for p in p_list:
        chinese_medicine_text += p.text + '\n'

    return chinese_medicine_text


def get_chinese_medicine_data():
    """
    获取中药数据
    :return: 中药数据
    :shape: dict: { '中药名': '中药记载、性质及推荐药方等' }
    """
    url = 'http://zhongyibaodian.com/bcgm/bencaogangmu-wanzhengban.html'
    Host = 'zhongyibaodian.com'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'Host': Host,
        'Connection': 'keep-alive'  # 保持连接
    }
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    a_list = soup.select('.post-body a')  # 选取class=post-body下的所有a标签
    del a_list[:7]
    chinese_medicine_data = {}
    # 获取中药名和中药文本的href
    for a in a_list:
        chinese_medicine_data[a.text] = a['href']  # shape: { '中药名': 'href' }
    # 选取前15种中药
    chinese_medicine_data = dict(list(chinese_medicine_data.items())[:15])
    # 将中药文本添加到中药数据中
    for name in chinese_medicine_data.keys():
        chinese_medicine_data[name] = get_chinese_meidicine_text(Host, chinese_medicine_data[name])

    return chinese_medicine_data  # shape: { '中药名': '中药记载、性质及推荐药方等' }


if __name__ == '__main__':
    data = get_chinese_medicine_data()
    df = pd.DataFrame.from_dict({'name': list(data.keys()), 'content': list(data.values())})
    # 将葳蕤替换成玉竹
    df.name = df.name.replace('葳蕤', '玉竹')
    # 删除长松，隐忍叶，荠，芦所在的行
    for name in ['长松', '隐忍叶', '荠', '芦']:
        df.drop(df[df.name == name].index, inplace=True)
    # 重置索引
    df.reset_index(drop=True, inplace=True)
    # 保存数据
    df.to_csv('../data/traditional_chinese_medicine_data.csv', index=False, encoding='utf-8-sig')
