import pandas as pd
import numpy as np
from sklearn import preprocessing

data_X = pd.read_csv('D:/Heart/pre/May.csv', encoding='gbk')

for column in list(data_X.columns[data_X.isnull().sum() > 0]):
    mean_val = data_X[column].mean()
    data_X[column].fillna(mean_val, inplace=True)

# 判断是否有空值
print(data_X.isnull().any())

# data_X = data_X.astype(int)

data_X['time'] = 7*(data_X['time']//10)+data_X['time']%10
# A = np.array(data_X)

min_max_scaler = preprocessing.MinMaxScaler()

# data_X.to_csv('D:/Heart/pre/May_pre.csv', header=False, index=False)

# data_normal = data_X[data_X['label'] == 0]
# data_normal = data_normal.drop('label', axis=1)
# data_normal = data_normal.astype("float")
# # print(data_normal)

# print(data_X)

np.set_printoptions(threshold=np.inf)

data_minmax = min_max_scaler.fit_transform(data_X)

print(data_minmax)
# data_normal_minmax = min_max_scaler.fit_transform(data_normal)
# print(data_normal_minmax)

# print(np.array(data_normal))

# data_abnormal = data_X[data_X['label'] == 1]
# print(data_abnormal)
# data_abnormal = data_abnormal.drop('label', axis=1)
# print(np.array(data_abnormal))
# data_abnormal_minmax = min_max_scaler.fit_transform(data_abnormal)

# print(np.array(data_abnormal_minmax))

# data_normal.to_csv('D:/Heart/pre/May_normal.csv', header=False, index=False)
# data_abnormal.to_csv('D:/Heart/pre/May_abnormal.csv', header=False, index=False)

