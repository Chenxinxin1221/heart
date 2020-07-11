import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score,precision_score,recall_score
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

data_X = pd.read_csv('D:/Heart/april_data.csv', encoding='gbk')   # 读取中文
X_label = pd.read_csv('D:/Heart/april_label.csv')

# 对训练集选取可用的特征
col = data_X.select_dtypes(exclude='object').columns
name = [a for a in col if a not in ['clinic number']]
# data_name = name[:-1]
# label_name = name[-1]
X_data = data_X[name]
# X_label = data_X[label_name]

X_data = X_data.fillna(X_data.mean())

op = X_data.sum(axis=1)    # 归一化
X_data = X_data.div(op, axis='rows')

pca = PCA(n_components=13)
X_data = pca.fit_transform(X_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, X_label, test_size=0.5, random_state=0)

# train kNN detector
clf_name = 'IForest'
clf = KNN()
# clf = IForest(n_estimators = 1000, contamination = 0.2, random_state=0)
clf.fit(X_train)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores

# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_pred)
