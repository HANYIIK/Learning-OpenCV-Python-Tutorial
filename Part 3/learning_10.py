# Chapter 22 直方图
"""
1、定义：图像中每个像素值的个数统计(比如说一副灰度图中像素值为 0 的有多少个，1 的有多少个)
2、函数：cv2.calcHist(), cv2.equalizeHist()
3、术语：
① dims：要计算的通道数，对于灰度图dims=1，普通彩色图dims=3；
② range：要计算的像素值范围，一般为[0,255];
③ bins：子区段数目。
    如果我们统计 0 ~ 255 每个像素值，bins=256；
    如果划分区间，比如 0 ~ 15, 16 ~ 31 … 240 ~ 255 这样16个区间，bins=16。
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


# 保存图像函数
def SaveOrNot(pic):
    if cv2.waitKey(0) == 27:  # wait for ESC to exit
        print('Not saved!')
        cv2.destroyAllWindows()
    elif cv2.waitKey(0) == ord('s'):  # wait for 's' to save and exit
        cv2.imwrite('result.jpg', pic)  # save
        print('Saved successfully!')
        cv2.destroyAllWindows()


# 显示图像函数
def ShowImage(name_of_image, image_, rate):
    img_min = cv2.resize(image_, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow(name_of_image, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_image, img_min)
    SaveOrNot(image_)


# 画图集函数
def DrawPictures(images_set, titles_set, rows, cols, num):
    for i in range(num):
        plt.subplot(rows, cols, i+1), plt.imshow(images_set[i]), plt.title(titles_set[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

img = cv2.imread('3.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1.1 OpenCV 中直方图计算
# cv2.calcHist(
#   ① 要计算的原图 - images,
#   ② 灰度图写[0]就行、彩色图B/G/R分别传入[0]/[1]/[2] - channels,
#   ③ 要计算的区域 - mask,
#   ④ 子区段数目 - histSize,
#   ⑤ 要计算的像素值范围 - ranges
# )
# hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# 非填充图
# plt.plot(hist)
# plt.show()

# 1.2 Numpy 中直方图计算
# hist, bins = np.histogram(img_gray.ravel(), 256, [0, 256])
# 填充图
# plt.hist(img_gray.ravel(), 256, [0, 256])
# plt.show()

# 2 直方图均衡化
# 一副效果好的图像通常在直方图上的分布比较均匀
# 直方图均衡化就是用来改善图像的全局亮度和对比度

# 2.1 普通均衡化
equ = cv2.equalizeHist(img_gray)

# 2.2 自适应均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img_gray)

ShowImage('普通均值化与自适应均衡化对比', np.hstack((equ, cl1)), 0.1)
