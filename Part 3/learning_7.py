# Chapter 19 Canny 边缘检测
import cv2
import numpy as np
from matplotlib import pyplot as plt


# 画图函数
def DrawPictures(images_set, titles_set, rows, cols, num):
    for i in range(num):
        plt.subplot(rows, cols, i+1), plt.imshow(images_set[i]), plt.title(titles_set[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

