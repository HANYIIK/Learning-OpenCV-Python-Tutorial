# Chapter 18 图像梯度
"""
三种梯度滤波器
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


# 画图函数
def DrawImage(images_set, titles_set, rows, cols, num):
    for i in range(num):
        plt.subplot(rows, cols, i+1), plt.imshow(images_set[i]), plt.title(titles_set[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# 18.1 Sobel 算子和 Scharr 算子: 用来求一阶或二阶􏱆导数
# Scharr 是对 Sobel􏰀（使用小的卷积核求解􏰎梯度􏱇角度时）􏰁的优化
print('test')

# 18.2 Laplacian 算子: 用来求二阶􏱆导数
