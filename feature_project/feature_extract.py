import re
import numpy as np
import pandas as pd

round1_data_select = 'data/data_select.csv'

data_select_form = pd.read_csv(round1_data_select, low_memory=False)
# print(data_select_form)

selectList = [
    'vid', '1814', '1815', '1840', '1850', '190', '191', '2302', '2403',
    '2404', '2405', '3190', '3191', '3192', '3193', '3195', '3196', '3197',
    '3430'
]

data_select = data_select_form[selectList].copy()
# print(data_select['1840'].head())

na_col = data_select.dtypes[data_select.isnull().any()]
# print(na_col)

# 离散型特征映射
mapper = {'2302': {'健康': 0, '亚健康': 1, '疾病': 2}}


def fuhao_map(fuhao):
    if fuhao == '阴性':
        return 1
    if fuhao == '阳性':
        return 5
    if fuhao == '未做':
        return 0
    # 数据中'+'个数
    num_plus = len(re.findall(re.compile(".*?(\+).*?"), fuhao))
    # 数据中'-'个数
    num_minus = len(re.findall(re.compile(".*?(\-).*?"), fuhao))
    # 只有一个'-'，无'+',与阴性结果相同，返回 1
    if num_minus >= 1 and num_plus == 0:
        return 1
    # 一个'+'，若干个'-'，返回2
    elif num_plus == 1 and num_minus >= 0:
        return 2
    # 两个'+'，若干个'-'，返回3
    elif num_plus == 2 and num_minus >= 0:
        return 3
    # 其他结果
    else:
        return 4


def chara_map(chara):
    chara = str(chara).replace('。', '.')
    # 匹配出数字，取第一个，因为数据中类似'14.0 14.0'的重复的脏数据
    reg_label = re.findall(re.compile('\d.*\d'), str(chara))
    # reg_label = str(chara).split(' ')[0]
    if reg_label:
        return reg_label[0].split(' ')[0]
    # 没有匹配出数字，说明是'未查' or'弃查'
    else:
        return '0'


def jiankang(chara):
    chara_ya = re.findall(re.compile('亚健康'), str(chara))
    if chara_ya:
        return '亚健康'
    elif re.findall(re.compile('健康'), str(chara)):
        return '健康'
    else:
        return '疾病'


for col in na_col.index:
    if col == '1814' or col == '1840' or col == '1850' or col == '190' or col == '191' or col == '3193':
        data_select[col] = data_select[col].apply(chara_map)
        data_select[col] = pd.DataFrame(data_select[col], dtype=np.float)
        med = data_select[col].mean()
        data_select[col].fillna(med, inplace=True)
    elif col == '1815' or col == '2403' or col == '2404' or col == '2405':
        data_select[col].fillna('0', inplace=True)
        data_select[col] = data_select[col].apply(chara_map)
        data_select[col] = list(map(float, data_select[col].values))
        med = data_select[col].mean()
        # print(med)
        data_select[col] = data_select[col].apply(
            lambda x: med if x == 0.00 else x)

    elif col == '2302':
        # 映射转换
        # print(data_select[col])
        data_select[col] = data_select[col].map(jiankang)
        for col, mapItem in mapper.items():
            data_select.loc[:, col] = data_select[col].map(mapItem)
        data_select[col].fillna(3, inplace=True)
    else:
        data_select[col].fillna('0', inplace=True)
        data_select[col] = data_select[col].apply(fuhao_map)

data_select['BMI'] = data_select['2403'] / ((data_select['2404'] / 100)**2)

print('测试集，训练集的列数为：{0}，行数为：{1}'.format(data_select.columns.size,
                                       data_select.iloc[:, 0].size))

data_select.to_csv(
    'data/feature_total_data_0415.csv', index=None, encoding="utf-8")
