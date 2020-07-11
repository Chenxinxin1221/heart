import numpy as np


# 定义两种激活函数（tanh,logistic）以及它们的导数
def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    # 构造函数
    def __init__(self, layers, activation='tanh'):
        '''
        : param layers: list类型,比如[2,2.1]代表输入层有两个神经元,隐藏层有两个，输出层有一个
        : param activation: 激活函数
        '''
        self.layers = layers
        # 选择后面用到的激活函数
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        # 定义网络的层数（是指总共有多少个输入层、隐层、输出层等）
        self.num_layers = len(layers)
        '''
        生成除输入层外的每层中神经元的biase（偏置）值，在（-1，1）之间，每一层都是一行一维数组数据
        randn函数执行一次生成x行y列的数据
        '''
        # 是根据除了输入层的节点来输入偏置
        self.biases = [np.random.randn(x) for x in layers[1:]]  # 偏置 x有几个数则生成几个具有标准正态分布的样本
        # print("偏置：", self.biases)  # 三个数组组成的列表
        '''
        随机生成每条连接线的权重，在（-1,1）之间
        weights[i-1]代表第i层和第i-1层之间的权重，元素个数等于i层神经元个数
        weights[i-1][0]表示第i层中第一个神经单元和第i-1层每个神经元的权重，元素个数等于i-1层神经元个数
        '''
        # layers[:-1]：第一个元素到倒数第二个元素，layers[1:]：第二个元素到最后一个元素
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]   # 反向是为了好进行矩阵运算
        # print("权重：", self.weights)

    # 训练模型，进行建模
    def fit(self, X, y, learning_rate=0.2, epochs=1):
        '''
        : param self: 当前对象指针
        : param X: 训练集
        : param y: 训练标记
        : param learning_rate: 学习率
        : param epochs: 训练次数
        : return: void
        '''
        for k in range(epochs):     # 循环epoch次
            for i in range(len(X)):      # 对4个样本进行循环
                # 存储本次的输入和后几层的输出
                # print(X)
                activations = [X[i]]
                # print(activations)
                # print(activations[-1])
                # 向前一层一层的走
                for b, w in zip(self.biases, self.weights):    # 对某个样本从输入到输出
                    # print("w:", w)
                    # print("activations[-1]:", activations[-1])  # 第一次得到[0 0]，第二次及后面是得到的上一次的输出值
                    # print("b:", b)
                    # 计算激活函数的参数,计算公式：权重.dot(输入)+偏向
                    z = np.dot(w, activations[-1]) + b
                    # 计算输出值（对每一层的输出进行激活函数）
                    output = self.activation(z)
                    # 将本次输出放进输入列表，后面更新权重的时候备用
                    activations.append(output)
                    # print(activations)

                error = y[i] - activations[-1]      # 计算误差值
                # print("实际y值:", y[i])
                # print("预测值：", activations[-1])
                # print("误差值", error)

                # 计算输出层误差率
                deltas = [error * self.activation_deriv(activations[-1])]

                # 循环计算隐藏层的误差率,从倒数第2层开始
                for i in range(self.num_layers - 2, 0, -1):     # l的值i依次为2,1（实际是第三层，第二层）
                    # print("第{0}层的权重:{1}".format(i, self.weights[i]))
                    # print("第{0}层的误差率:{1}".format(i+1, deltas[-1]))
                    # print(activations[i])
                    # 计算出的误差率在下一次循环的时候又要使用
                    # print(self.activation_deriv(activations[i]))
                    # print(np.dot(deltas[-1], self.weights[i]))
                    # print(self.activation_deriv(activations[i]) * np.dot(deltas[-1], self.weights[i]))
                    deltas.append(self.activation_deriv(activations[i]) * np.dot(deltas[-1], self.weights[i]))  # *是对应直接相乘

                # 将各层误差率顺序颠倒，准备逐层更新权重和偏向
                deltas.reverse()  # 这是列表的方法
                # print "每层的误差率：",deltas

                # 根据每一次层的误差率，我们可以使用权重的更新公式和偏向的更新公式更新权重和偏向。
                # 更新权重和偏置
                for j in range(self.num_layers - 1):
                    # 本层结点的输出值
                    layers = np.array(activations[j])
                    # print "本层输出：",layers
                    # print "错误率：",deltas[j]
                    # 权重的增长量，计算公式，增长量 = 学习率 * (错误率.dot(输出值))
                    delta = learning_rate * ((np.atleast_2d(deltas[j]).T).dot(np.atleast_2d(layers)))
                    # 更新权重
                    self.weights[j] += delta
                    # print "本层偏向：",self.biases[j]
                    # 偏向增加量，计算公式：学习率 * 错误率
                    delta = learning_rate * deltas[j]
                    # print np.atleast_2d(delta).T
                    # 更新偏向
                    self.biases[j] += delta
                # print self.weights

    def predict(self, x):
        '''
        :param x: 测试集
        :return: 各类型的预测值
        '''
        for b, w in zip(self.biases, self.weights):
            # 计算权重相加再加上偏向的结果
            z = np.dot(w, x) + b
            # 计算输出值
            x = self.activation(z)
        return x


# nn = NeuralNetwork([2,4,3,1], 'tanh')   # 初始化（对应init函数进行书写）
# # 训练集
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# # lanbel标记
# y = np.array([0, 1, 1, 0])
# # 建模
# nn.fit(X, y, epochs=1000)
# # 预测
# for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
#     print(i, nn.predict(i))