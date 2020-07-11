from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.inf)  # 输出数组中的全部元素

# # 图片灰度化
# # 彩色（三维）---->灰度（二维）
# img = Image.open("image/strawberry.jpg")  # 打开一张图片
# print(img)
# print(img.shape)    # (417, 500, 3) 高、宽、通道数
# # img.show()   # 彩色
# img = img.convert("L")  # 图片灰度化
# img = np.asarray(img)
# print(img.shape)
# # img.show()   # 显示图片（灰度图）
# print(np.asarray(img))  # 将图片转换为数组形式，元素为其像素的亮度值


# 图片二值化
img = Image.open("image/3.jpg")  # 打开一张图片
img = img.convert("L")  # 图片灰度化
# print(np.array(img))
# point函数是用来二值化图片的，其参数是一个lambda函数：以120为阈值，大于120为1，小于等于120为0
img = img.resize((32, 32))   # 缩放图片，不是直接剪辑
img = img.point(lambda x: 1 if x > 120 else 0)
img_array = np.asarray(img)   # 用1和0画出了3
# img.show()
# print(img_array)


# 获取网格特征数字统计图
# 我们的图片尺寸是32*32的，所以我们将其化为8*8的点阵图,步骤如下：
# 1、将二值化后的点阵水平平均划线分成8份，竖直平均划线分成8份。
# 2、分别统计每一份中像素点为1的个数。
# 3、将每一个份统计值组合在一起，构成8*8的点阵统计图

# 将二值化后的数组转化成网格特征统计图
def get_features(array):
    h, w = array.shape  # 得到数组的高度和宽度 (32, 32)
    data = []   # data是我们最后得到的数组（网格特征统计图）
    # print(h//4) # 得到int类型
    for x in range(0, w//4):   # x循环宽度方向0-7  在python
        offset_y = x * 4  # 每4行进行计算
        temp = []
        for y in range(0, h//4):  # y循环高度方向0-7
            offset_x = y * 4      # 在某四行跨四列进行计算
            # 统计每个区域的1的值
            temp.append(sum(sum(array[0+offset_y:4+offset_y, 0+offset_x:4+offset_x])))  # 每四行的8个和
        data.append(temp)
        # print(data)
    return np.asarray(data)


# 得到网格特征统计图
features_array = get_features(img_array)
print(features_array)
# 将二维统计图转换为一维特征向量（直接把它们都放到一行）
# 一个样例转换为一行特征向量，多个样例组成一个数据集
features_vector = features_array.reshape(features_array.shape[0]*features_array.shape[1])
print(features_vector)