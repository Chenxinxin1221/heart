import pandas as pd
import numpy as np
import warnings

from sklearn import preprocessing
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

from sklearn.metrics import f1_score,precision_score,recall_score

warnings.filterwarnings('ignore')

data_X = pd.read_csv('D:/Heart/april_data.csv', encoding='gbk')   # 读取中文
X_label = pd.read_csv('D:/Heart/april_label.csv')

# 对训练集选取可用的特征
# print(data_X.dtypes)
col = data_X.select_dtypes(exclude='object').columns
name = [a for a in col if a not in ['clinic number']]
# data_name = name[:-1]
# label_name = name[-1]
X_data = data_X[name]
print(X_data)
# X_label = data_X[label_name]


# # 用每列的均值填充缺失值
X_data = X_data.fillna(X_data.mean())
# print(X_data)

op = X_data.sum(axis=1)     # .sum(axis=1)计算每一行向量之和
X_data = X_data.div(op, axis='rows')
# print(X_data)

# print(np.isnan(X_data).any())  # 判断是否还有缺失值

pca = PCA(n_components=14)
X_data = pca.fit_transform(X_data)
# print(X_data)

# 归一化
# min_max_scaler = preprocessing.MinMaxScaler()
# X_data = min_max_scaler.fit_transform(X_data)
# print(X_data)


X_train, X_test, Y_train, Y_test = train_test_split(X_data, X_label, test_size=0.3, random_state=1)
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

# 拟合模型
model = XGBClassifier(
    # 树的个数
    n_estimators=30,
    # 如同学习
    learning_rate=0.02,
    # 构建树的深度，越大越容易过拟合
    max_depth=8,
    # 随机采样训练样本 训练实例的子采样比
    subsample=1,
    # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子
    gamma=0.1,
    # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    eeg_lambda=2,
    # 最大增量步长，我们允许每个树的权重估计。
    max_delta_step=0,
    # 生成树时进行的列采样
    colsample_bytree=1,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # 假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    min_child_weight=1,
    #随机种子
    seed=0,
    # L1 正则项参数
    reg_alpha=0,
    #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
    scale_pos_weight=0,
    #多分类的问题 指定学习任务和相应的学习目标
    objective= 'multi:softmax',
    # 类别数，多分类与 multisoftmax 并用
    num_class=2,
    # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
    silent=0,
    # cpu 线程数 默认最大
    nthread=4)

# param_grid = {
#         'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2],
#     }
# model = GridSearchCV(model, param_grid)  # 参数调优

model.fit(X_train, Y_train)

# 预测
y_pred = model.predict(X_test)

# 评估结果
r = recall_score(Y_test, y_pred)    # 医学看中召回率
p = precision_score(Y_test, y_pred)

print("recall_score:", r)
print("precision_score:", p)

# 天池：二手车预价
# xgboost处理心脏数据，召回率0.14，精确率0.5
# 最近在学习深度学习方面的东西