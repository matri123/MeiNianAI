# coding: utf-8
import pandas as pd
import numpy as np
import re
import math
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def train_map(label):
    if label == u'弃查' or label == u'未查':
        return '0'
    else:
        return label

def re_match(x):
    if x == u'7.75轻度乳糜':
        return '7.75'
    reg_label = re.findall(re.compile('\d+\.\d+'), str(x))
    if reg_label:
        return reg_label[0]
    else:
        return x

def keycount(ll):
    mm = {}
    for i in ll:
        if i in mm:
            mm[i] += 1
        else :
            mm[i] = 1
    return mm


def deal_train_label(train_label):
    train_label[u'收缩压'] = train_label[u'收缩压'].apply(train_map)
    train_label[u'收缩压'] = train_label[u'收缩压'].apply(
        lambda x: '126.03' if x == '0' else x)
    train_label[u'舒张压'] = train_label[u'舒张压'].apply(train_map)
    train_label[u'舒张压'] = train_label[u'舒张压'].apply(
        lambda x: '79.63' if x == '0' else x)
    train_label[u'血清甘油三酯'] = train_label[u'血清甘油三酯'].apply(train_map)
    train_label[u'血清高密度脂蛋白'] = train_label[u'血清高密度脂蛋白'].apply(train_map)
    # train_label['血清低密度脂蛋白'] = train_label['血清低密度脂蛋白'].apply(train_map)

    # train_label['收缩压'] = train_label['收缩压'].apply(re_match)
    # train_label['舒张压'] = train_label['舒张压'].apply(re_match)
    train_label[u'血清甘油三酯'] = train_label[u'血清甘油三酯'].apply(re_match)
    # train_label['血清高密度脂蛋白'] = train_label['血清高密度脂蛋白'].apply(re_match)
    # 标签中有负数，需要转化，否则在训练时会出错，因为使用了'neg_mean_squared_log_error'评价指标，预测值不能为负数
    train_label[u'血清低密度脂蛋白'] = train_label[u'血清低密度脂蛋白'].apply(
        lambda x: abs(float(x)) if float(x) < 0 else x)

    train_label_shousuo = list(map(float, train_label[u'收缩压'].values))
    train_label_shuzhang = list(map(float, train_label[u'舒张压'].values))
    train_label_ganyousanzhi = list(map(float, train_label[u'血清甘油三酯'].values))
    train_label_gaomiduzhidanbai = list(
        map(float, train_label[u'血清高密度脂蛋白'].values))
    train_label_dimiduzhidanbai = list(
        map(float, train_label[u'血清低密度脂蛋白'].values))

    return train_label_shousuo, train_label_shuzhang, train_label_ganyousanzhi, train_label_gaomiduzhidanbai, train_label_dimiduzhidanbai

def train_nn(m, trainX, trainY, epoch=3000, lr=1e-3):
    adam = Adam(lr)
    m.compile(loss='mse', optimizer=adam)
    for i in xrange(0, epoch):
        loss = m.train_on_batch(trainX, trainY)
        print('iter:%d loss:%f'%(i, loss))

def test_nn(m, testX):
    return m.predict(testX)

def init_nn():
    model = Sequential()
    model.add(Dense(100, input_dim=81, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    return model


if __name__ == '__main__':
    print("***************************开始导入数据集*******************************")
    feature_train_data = pd.read_csv('data/selected_train_data.csv')
    feature_test_data = pd.read_csv('data/selected_test_data.csv')
    train_label = pd.read_csv(
        'data/meinian_round1_train_20180408.csv', encoding='gbk')
    test_label = pd.read_csv(
        'data/meinian_round1_test_a_20180409.csv', encoding='gbk')

    print('训练集的列数为：{0}，行数为：{1}'.format(feature_train_data.columns.size,
                                       feature_train_data.iloc[:, 0].size))
    print('测试集的列数为：{0}，行数为：{1}'.format(feature_test_data.columns.size,
                                       feature_test_data.iloc[:, 0].size))
    # print(feature_train_data.columns)

    X_train = feature_train_data.drop('vid', axis=1)
    X_test = feature_test_data.drop('vid', axis=1)

    test_label.drop(
        [u'收缩压', u'舒张压', u'血清甘油三酯', u'血清高密度脂蛋白', u'血清低密度脂蛋白'], axis=1, inplace=True)
    # print(test_label)
    # print(train_label['血清低密度脂蛋白'].isnull().any())
    # print(train_label['收缩压'].value_counts())
    print("***********************开始处理训练集标签脏数据*************************")
    train_label_shousuo, train_label_shuzhang, train_label_ganyousanzhi, train_label_gaomiduzhidanbai, train_label_dimiduzhidanbai = deal_train_label(
        train_label)
    # print(np.nan in train_label_dimiduzhidanbai)
    # print(train_label[train_label['血清甘油三酯'] == '2.2.8'])
    print("训练集标签脏数据处理完毕!")
    print("****************************开始训练模型********************************")
    model = init_nn()
    train_nn(model, X_train, train_label_shousuo)
    test_label['shousuoya'] = test_nn(model, X_test)
    x_pre_shousuo = test_nn(model, X_train)

    model = init_nn()
    train_nn(model, X_train, train_label_shuzhang)
    test_label['shuzhangya'] = test_nn(model, X_test)
    x_pre_shuzhang = test_nn(model, X_train)

    model = init_nn()
    train_nn(model, X_train, train_label_ganyousanzhi)
    test_label['ganyousanzhi'] = test_nn(model, X_test)
    x_pre_ganyou = test_nn(model, X_train)

    model = init_nn()
    train_nn(model, X_train, train_label_gaomiduzhidanbai)
    test_label['gaomiduzhidanbai'] = test_nn(model, X_test)
    x_pre_gaomi = test_nn(model, X_train)

    model = init_nn()
    train_nn(model, X_train, train_label_dimiduzhidanbai)
    test_label['dimiduzhidanbai'] = test_nn(model, X_test)
    x_pre_dimi = test_nn(model, X_train)

    test_label.to_csv('./data/ans.csv', index=False, header=None)

    shous, shus, gans, gaos, dis = 0,0,0,0,0
    for i in xrange(len(x_pre_shousuo)):
        shous += (math.log(x_pre_shousuo[i] + 1) - math.log(train_label_shousuo[i] + 1)) ** 2
        shus += (math.log(x_pre_shuzhang[i] + 1) - math.log(train_label_shuzhang[i] + 1)) ** 2
        if x_pre_ganyou[i] < -1:
            x_pre_ganyou[i] = 0
        gans += (math.log(x_pre_ganyou[i] + 1) - math.log(train_label_ganyousanzhi[i] + 1)) ** 2
        gaos += (math.log(x_pre_gaomi[i] + 1) - math.log(train_label_gaomiduzhidanbai[i] + 1)) ** 2
        dis += (math.log(x_pre_dimi[i] + 1) - math.log(train_label_dimiduzhidanbai[i] + 1)) ** 2
    leng = len(x_pre_shousuo)
    loss = (shous + shus + gans + gaos + dis) / leng / 5
    print(loss)

    # print(X_train)
    # xgb_cv(X_train, train_label_shousuo)
    # xgb_cv(X_train, train_label_shuzhang)
    # xgb_cv(X_train, train_label_ganyousanzhi)
    # xgb_cv(X_train, train_label_gaomiduzhidanbai)
    # xgb_cv(X_train, train_label_dimiduzhidanbai)
    print("****************************开始预测结果********************************")
    # model_process(X_train, train_label_shousuo, train_label_shuzhang,
    #               train_label_ganyousanzhi, train_label_gaomiduzhidanbai,
    #               train_label_dimiduzhidanbai, X_test)
    print(
        "*******************************程序结束*********************************")
