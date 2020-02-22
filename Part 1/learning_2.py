# 绘图
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('1.jpg', 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # 隐藏 x 坐标轴与 y 坐标轴
plt.show()
