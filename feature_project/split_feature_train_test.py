import pandas as pd

feature_total_data_path = 'data/feature_total_data_0415.csv'
train_labe_path = 'data/meinian_round1_train_20180408.csv'
test_label_path = 'data/meinian_round1_test_a_20180409.csv'

feature_total_data = pd.read_csv(feature_total_data_path)

train_label = pd.read_csv(train_labe_path, encoding='gbk')

test_label = pd.read_csv(test_label_path, encoding='gbk')

train_id = train_label['vid'].tolist()

test_id = test_label['vid'].tolist()

train_data = feature_total_data[(feature_total_data['vid'].isin(train_id))]

test_data = feature_total_data[(feature_total_data['vid'].isin(test_id))]

train_data.to_csv('data/feature_train_data_0415.csv', index=False)

print('训练集的列数为：{0}，行数为：{1}'.format(train_data.columns.size,
                                   train_data.iloc[:, 0].size))

test_data.to_csv('data/feature_test_data_0415.csv', index=False)

print('测试集的列数为：{0}，行数为：{1}'.format(test_data.columns.size,
                                   test_data.iloc[:, 0].size))
