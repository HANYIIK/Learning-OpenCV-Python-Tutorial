# Chapter 9 图像的基础操作
import cv2
from matplotlib import pyplot as plt

# import numpy as np
# numpy 是经过优化了的进行快速矩阵运算的软件包

# 9.1 根据图像行和列的坐标获取像素的BGR值
print('根据图像行和列的坐标获取像素的BGR值')
img = cv2.imread('1.jpg')
px = img[100, 100]
print(px)
blue = img[100, 100, 0]  # 显示(100, 100)坐标处像素的B值
green = img[100, 100, 1]  # 显示(100, 100)坐标处像素的G值
red = img[100, 100, 2]  # 显示(100, 100)坐标处像素的R值
print(blue)
print(green)
print(red)

# 修改像素的BGR值
print('\n修改像素的BGR值')
img[100, 100] = [255, 255, 255]
print(img[100, 100])

# 用numpy的矩阵修改像素的BGR值
print('\n用numpy的矩阵操作像素')
print(img.item(100, 100, 2))  # 显示(100, 100)坐标处像素的R值
img.itemset((100, 100, 2), 100)  # 修改(100, 100)坐标处像素的R值为100
print(img.item(100, 100, 2))

# 9.2 获取图像属性
print('\n图像属性')
print(img.size)  # 像素的数目
print(img.shape)  # 返回一个(行数, 列数, 通道数)的元祖
print(img.dtype)  # 返回图像的数据类型

# 9.3 图像切割与局部替换(ROI技术)
ball = img[1:1500, 1:1500]
img[1501:3000, 1501:3000] = ball

# 9.4 拆分及合并图像通道
'''
拆分及合并图像通道：
    ① 对BGR三个通道分别进行操作: 把BGR拆分成单个通道
        b, g, r = cv2.split(img)
    ② 把独立通道的图片合并成一个BGR图像
        img = cv2.merge(b, g, r)
'''
# img[:, :, 2] = 0  # 令红色通道为0

# 9.5 为图像扩充边界
BLUE = [255, 0, 0]  # 定义 matplotlib 的蓝色
'''
 【注意】
    matplotlib 输出图像是'R G B'顺序
    而 OpenCV 输出图像是'B G R'顺序
'''

replicate = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=BLUE)

plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
plt.show()
