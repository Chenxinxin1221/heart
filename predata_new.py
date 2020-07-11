import pandas as pd
import numpy as np
from sklearn import preprocessing

data_X = pd.read_csv('D:/Heart/pre/May.csv', encoding='gbk')

# print(data_X.isnull().any())
# print(list(data_X.columns[data_X.isnull().sum() > 0]))
# 给空值赋值
for column in list(data_X.columns[data_X.isnull().sum() > 0]):
    mean_val = data_X[column].mean()
    data_X[column].fillna(mean_val, inplace=True)

# print(data_X.isnull().any())

data_X['time'] = 7*(data_X['time']//10)+data_X['time']%10

# print(data_X)

min_max_scaler = preprocessing.MinMaxScaler()
data_sklearn = min_max_scaler.fit_transform(data_X)
# print(data_sklearn)

# data_minmax = (data_X - data_X.min(axis=0)) / (data_X.max(axis=0) - data_X.min(axis=0))
# print(data_minmax)

data_minmax = pd.DataFrame(data_sklearn)
# print(data_minmax)

np.set_printoptions(threshold=np.inf)
# print(data_minmax)
# data_abnormal = data_minmax[data_minmax[15] == 1]
# data_abnormal = data_abnormal.drop(15, axis=1)
# print(np.array(data_abnormal))
# data_abnormal = np.array(data_abnormal)
data_normal = data_minmax[data_minmax[15] == 0]
data_normal = data_normal.drop(15, axis=1)