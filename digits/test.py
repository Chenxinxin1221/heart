from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
import numpy as np
# from BpNN import NeuralNetwork
from sklearn.datasets import load_digits    # 手写数字数据集

# # 加载数据集并进行数据预处理
# dataset = load_digits()
# # print(dataset)
# # print(dataset.data)    # 训练集：手写数字特征向量数据集，每一个元素都是一个64维的特征向量 （1797*64）
# # print(dataset.target)  # 标签：特征向量对应的标记，每一个元素都是自然是0-9的数字 （1797*1）
# # print(dataset.images)  # 对应着data中的数据，每一个元素都是8*8的二维数组，其元素代表的是灰度值，转化为一维时便是特征向量 （1797, 8, 8）
# # 训练集
# X = dataset.data
# # 标记
# Y = dataset.target
# # 数据与处理，让特征值都处在0-1之间
# X -= X.min()
# X /= X.max()
# # 切分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
# # 对标记进行二值化
# labels_train = LabelBinarizer().fit_transform(y_train)

# x = np.arange(16).reshape(4, 4)
# print(x)
# print(x[0:2, 0:3])
# print(sum(x[0:2, 0:3]))
# print(sum(sum(x[0:2, 0:3])))

# a = [1,2,3,9,12,4]
# # print(len(a))
# print(a[1:])  # [2, 3, 9, 12, 4]
# # for i in a[1:]:
# #     print(i)
# print(a[:-1])  # [1, 2, 3, 9, 12]
# # print(zip(a[1:],a[:-1]))
# for x,y in zip(a[1:],a[:-1]):
#     print(x,y)

# X = np.array([[0, 0], [0, 1], [1, 0],[1, 1]])
# print(len(X))
# b = X[1]
# print(b)
# print(b[-1])
# print([b[-1]])
# print([b[-1]][-1])

# x = np.arange(10)
# x = list(x)
# print(x)
# x.reverse()   # reverse()是列表的方法，返回值为none
# print(x)


# for i in range(4 - 2, 0, -1):
#     print(i)

# a = np.array([1,2,3])
# b = np.array([1,0,2])
# print(a)
# print(b)
# print(a*b)   # 对应相乘
# print(np.dot(a, b))   # 矩阵乘


class Solution(object):
    result_list = []
    def generateParenthesis(self, n):
        global result_list
        def generate(s):
            global result_list
            new_s = s
            left = right = 0
            for i in range(n * 2):
                if s[i] == '(':
                    left += 1
                else:
                    right += 1
                    if left > right and s[i - 1] == '(':
                        new_s = s[:i - 1] + s[i:i + 1] + s[i - 1:i] + s[i + 1:]
                        if new_s not in result_list:
                            result_list += [new_s]
                            generate(new_s)
        s = '(' * n + ')' * n
        result_list = [s]
        generate(s)
        return result_list


print(Solution().generateParenthesis(3))
