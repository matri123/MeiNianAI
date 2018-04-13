import re
import numpy as np
import pandas as pd

round1_data_select = 'data/data_select_new.csv'

data_select_form = pd.read_csv(round1_data_select, low_memory=False)
# print(data_select_form)

selectList = [
    'vid', '0102', '0113', '0114', '0117', '1001', '1814', '1815', '1840',
    '1850', '190', '191', '2302', '2403', '2404', '2405', '3190', '3191',
    '3192', '3193', '3195', '3196', '3197', '3430', '0101', '0115', '0409',
    '0421', '0426', '10004'
]

data_select = data_select_form[selectList].copy()

# print(data_select['0102'].head())
# print(data_select['300005'].value_counts())

# na_col = data_select.dtypes[data_select.isnull().any()]
# print(na_col)

# 离散型特征映射
mapper = {'2302': {'健康': 0, '亚健康': 1, '疾病': 2}}


def fuhao_map(fuhao):
    if fuhao == '阴性' or fuhao == '正常' or fuhao == 'Normal' or fuhao == 'NormaL':
        return 1
    if fuhao == '阳性':
        return 5
    if fuhao == '未做':
        return 0
    # 数据中'+'个数
    num_plus = len(re.findall(re.compile(".*?(\+).*?"), fuhao))
    # 数据中'-'个数
    num_minus = len(re.findall(re.compile(".*?(\-).*?"), fuhao))
    # 只有若干个'-'，无'+',与阴性结果相同，返回 1
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


def map_1001(des):
    des = str(des)
    if '不齐' in des or '过缓' in des or '左室高电压' in des or '左偏' in des or '右偏' in des or '低平' in des or '阻滞' in des:
        return -1
    else:
        return 0


def map_0113(des):
    des = str(des)
    if '密集弥漫性增强' in des or '欠清晰' in des:
        return -1
    else:
        return 0


def map_0114(des):
    des = str(des)
    if '囊壁毛糙' in des or '较强回声附着' in des:
        return -1
    else:
        return 0


# 可能有浮点型，需要类型转换
def map_0117(des):
    des = str(des)
    if '无回声区' in des or '强回声' in des:
        return -1
    else:
        return 0


feature_list_one = [
    '1814', '1815', '1840', '1850', '190', '191', '2302', '2403', '2404',
    '2405', '3190', '3191', '3192', '3193', '3195', '3196', '3197', '3430'
]
for col in feature_list_one:
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

feature_list_two = ['0102', '0113', '0114', '0117', '1001']

for col in feature_list_two:
    if col == '1001':
        data_select[col] = data_select[col].apply(map_1001)
    elif col == '0102':
        data_select[col] = data_select[col].apply(
            lambda x: 1 if '脂肪肝' in str(x) else 0)
    elif col == '0113':
        data_select[col] = data_select[col].apply(map_0113)
    elif col == '0114':
        data_select[col] = data_select[col].apply(map_0114)
    elif col == '0117':
        data_select[col] = data_select[col].apply(map_0117)
    else:
        pass

feature_list_three = ['0101', '0115', '0409', '0421', '0426', '10004']


def jiazhuangxian_map(des):
    if isinstance(des, float):
        return des
    elif '分布均匀' in des or '未见明显异常回声' in des:
        return 0
    else:
        return -1


def yixian_map(des):
    if isinstance(des, float):
        return des
    elif '回声均匀' in des or '未见扩张' in des:
        return 0
    else:
        return -1


def bingshi_map(des):
    if isinstance(des, float):
        return des
    elif '病史' in des or '心动' in des or '腹壁' in des:
        return -1
    else:
        return 0


def xinlv_map(des):
    if isinstance(des, float):
        return des
    elif '早搏' in des or '不齐' in des or '过速' in des:
        return -1
    else:
        return 0


def dongmai_map(des):
    if isinstance(des, float):
        return des
    elif '收缩期杂音' in des:
        return -1
    else:
        return 0


def chongfu_map(chara):
    #  '4.14.'    '18.00 18.00'
    tmp = re.findall(re.compile('\d.*\d'), str(chara))
    # ???????????????不太理解
    if tmp:
        return tmp[0].split(' ')[0]
    else:
        return -1


for col in feature_list_three:
    if col == '10004':
        data_select[col] = data_select[col].apply(chongfu_map)
        data_select[col] = pd.DataFrame(data_select[col], dtype=np.float)
        med = data_select[col].mean()
        data_select[col].fillna(med, inplace=True)
    elif col == '0101':
        data_select[col] = data_select[col].apply(jiazhuangxian_map)
    elif col == '0115':
        data_select[col] = data_select[col].apply(yixian_map)
    elif col == '0409':
        data_select[col] = data_select[col].apply(bingshi_map)
    elif col == '0421':
        data_select[col] = data_select[col].apply(xinlv_map)
    elif col == '0426':
        data_select[col] = data_select[col].apply(dongmai_map)
    else:
        pass

print('测试集，训练集的列数为：{0}，行数为：{1}'.format(data_select.columns.size,
                                       data_select.iloc[:, 0].size))

data_select.to_csv(
    'data/feature_total_data_0413.csv', index=None, encoding="utf-8")