from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits    # 这个手写数字数据集没有图片，而是经过提取得到的手写数字特征和标签
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.externals import joblib
import numpy as np
import warnings
from BpNN import NeuralNetwork
np.set_printoptions(threshold=np.inf)  # 输出数组中的全部元素
warnings.filterwarnings("ignore")

# 加载数据集并进行数据预处理
###################使用sklearn中的数据集################
# 加载数据集
digits = load_digits()
# 手写数字特征向量数据集，每一个元素都是一个64维的特征向量。
X = digits.data
# print(X)
# print(X.shape)  # (1797, 64)
# 特征向量对应的标记，每一个元素都是自然是0-9的数字。
Y = digits.target
# print(Y)
# print(Y.shape)  # 1797*1
# 数据与处理，让特征值都处在0-1之间
X -= X.min()
X /= X.max()
# print(X)
# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
# print(y_train)
# 对标签进行二值化,因为神经网络只认识0和1
# 0000000000代表数字0, 0100000000代表数组1, 0010000000代表数字2 等
labels_train = LabelBinarizer().fit_transform(y_train)  # 对标签进行二值化
# print(labels_train)


# 构造神经网络模型
###########构造神经网络模型################
# 构建神经网络结构
nn = NeuralNetwork([64, 100, 10], 'logistic')   # 输出层64个神经元（因为特征为64），隐层100个神经元，输出层10个神经元（因为分10类）
# 训练模型
nn.fit(X_train, labels_train, learning_rate=0.2, epochs=200)
# # 保存模型
# joblib.dump(nn, 'model/nnModel.m')    # 将模型保存到本地，若没有model文件夹，需要手动创建
# # 加载模型
# nn = joblib.load('model/nnModel.m')   # 模型从本地调回


# 数字识别和模型测评
###############数字识别####################
# 存储预测结果
predictions = []
# 对测试集进行预测
for i in range(y_test.shape[0]):
    out = nn.predict(X_test[i])
    predictions.append(np.argmax(out))  # 取出out中元素最大值所对应的索引，那个索引对应的类则是我们预测的类

###############模型评估#####################
# 打印预测结果混淆矩阵
# confusion_matrix 混淆矩阵
# 混淆矩阵是机器学习中总结分类模型预测结果的情形分析表
# 以矩阵形式将数据集中的记录按照真实的类别与分类模型作出的分类判断两个标准进行汇总。
print(confusion_matrix(y_test, predictions))   # 所以对角线上的值越多越好

# 打印预测报告
# 列表左边的一列为分类的标签名，右边support列为每个标签的出现次数，avg / total行为各列的均值（support列为总和）．
# F1 值是精确度和召回率的调和平均值
print(classification_report(y_test, predictions))