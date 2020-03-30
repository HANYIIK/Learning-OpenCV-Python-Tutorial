# 机器学习
# Chapter 1 k 近邻(kNN)
"""
原理: 检测 k 个最近邻居的分类方法, 监督学习。
步骤:
① knn = cv2.ml.KNearest_create()
② knn.train(训练集, cv2.ml.ROW_SAMPLE, 训练集的监督值)
③ ret, 最终测试结果, k 个邻居的监督值, 距离的平方(int 型)集 = knn.findNearest(测试数据, k 值)
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# 一、生成红蓝家族地图
# 1.包含 25 个已知数据 / 训练数据的(x, y)值的特征集
trainData1 = np.random.randint(0, 60, (50, 2)).astype(np.float32)
trainData2 = np.random.randint(40, 100, (50, 2)).astype(np.float32)
trainData = np.vstack((trainData1, trainData2))

# 2.用数字 0 和 1 把 trainData 分别标记为红色和蓝色
response1 = np.random.randint(0, 1, (50, 1)).astype(np.float32)
response2 = np.random.randint(1, 2, (50, 1)).astype(np.float32)
response = np.vstack((response1, response2))

# 3.把红色 / 蓝色提取分类
red = trainData[response.ravel() == 0]      # 0 ---> red
blue = trainData[response.ravel() == 1]     # 1 ---> blue

# 4.画出地图
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')

# 5.画出测试点
newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)
plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o')

# 二、kNN 算法分类器初始化
knn = cv2.ml.KNearest_create()

# 三、kNN 算法
knn.train(trainData, cv2.ml.ROW_SAMPLE, response)
ret, results, neighbours, dist = knn.findNearest(newcomer, 20)

# 画出该点与近邻的直线
for distance in dist.ravel():
    for train_data in trainData:
        dis_test = int(math.pow(train_data[0] - newcomer[0][0], 2) +
                       math.pow(train_data[1] - newcomer[0][1], 2))
        if dis_test == int(distance):
            plt.plot([newcomer[0][0], train_data[0]],
                     [newcomer[0][1], train_data[1]])

plt.show()

if int(results.ravel()[0]) == 0:
    results = 'red'
else:
    results = 'blue'

print('result:', results)
