import pandas as pd
import numpy as np
import warnings

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

from sklearn.metrics import f1_score,precision_score,recall_score

warnings.filterwarnings('ignore')

# train = open(r'D:/Heart/heart data.csv')    # 这样才允许读取中文
# Train_data = pd.read_csv(train)
data_X = pd.read_csv('D:/Heart/april_data.csv', encoding='gbk')   # 读取中文
label_Y = pd.read_csv('D:/Heart/april_label.csv')   # 读取中文
# print(data_X.shape)
# print(data_X.head())
# print(label_Y.shape)
# print(label_Y.head())

# 对训练集选取可用的特征
# print(data_X.dtypes)
col = data_X.select_dtypes(exclude='object').columns
name = [a for a in col if a not in ['clinic number']]
# print(name)
X_data = data_X[name]
# print(X_data)

# # 用每列的均值填充缺失值
X_data = X_data.fillna(X_data.mean())
# print(X_data)

# print(np.isnan(X_data).any())  # 判断是否还有缺失值

# pc = PCA(n_components=12)
# X_data = pc.fit_transform(X_data)
# print(X_data)

# 归一化
min_max_scaler = preprocessing.MinMaxScaler()
X_data = min_max_scaler.fit_transform(X_data)
# print(X_data)

X_train, X_test, Y_train, Y_test = train_test_split(X_data, label_Y, test_size=0.4)
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

# 拟合模型
model = XGBClassifier()
model.fit(X_train, Y_train)

# 预测
y_pred = model.predict(X_test)

# 评估结果
r = recall_score(Y_test, y_pred)    # 医学看中召回率

print("recall_score:", r)