# Chapter 19 Canny 边缘检测
"""
Canny边缘检测:
cv2.Canny(img, 最低阈值, 最高阈值)
"""
import cv2
from matplotlib import pyplot as plt


# 画图集函数
def DrawPictures(images_set, titles_set, rows, cols, num):
    for i in range(num):
        plt.subplot(rows, cols, i+1), plt.imshow(images_set[i], cmap='gray'), plt.title(titles_set[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

img = cv2.imread('13.png', 0)
edges = cv2.Canny(img, 30, 70)

images = [img, edges]
titles = ['Original', 'Canny']

DrawPictures(images, titles, 1, 2, 2)
