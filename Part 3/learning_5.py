# Chapter 17 形态学转换
"""
根据图像的形状􏰱􏰲进行简单操作，一般情况下对【二值化图像】􏰱进行操作。
输入两个参数􏰊，第一个是原始图像，􏰊第二个􏰌称为结构化元素或核，􏰊它是用来决定操作的性性质的。
两个基本的形态学操作是【腐蚀】􏰷和【膨胀】。他们的变体构成了【开运算】、【闭运算】、􏰊【梯度】等。
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

# 17.1 腐蚀: 把前景物体的边􏰋界腐􏰷掉，􏰀但前景仍然是白色􏰁
# 卷积核沿着图像滑动􏰊, 如果与卷积核对应的原图像的所有像素值􏰣是 1, 那􏰊􏰤么中心元素就保持原来的像素值􏰊，否则就变为􏰽零。
# 用来【断开】两个连在一起的物体。
img = cv2.imread('j.png')

kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)

# 17.2 膨胀: 与腐􏰷相反􏰊, 与卷积核对应的原图像的像素值中只􏰐有一个是1, 􏰊中心元素的像素值就是1。
# 用来􏰿【连接】两个分开的物体。
dilation = cv2.dilate(img, kernel, iterations=1)

# 17.3 开运算: cv2.morphologyEx() 先􏰱进行腐蚀，􏰷再进行􏰱􏰲膨胀，就叫做开运算
# 用来去除噪声
# 去白
img_points = cv2.imread('points.png')
opening = cv2.morphologyEx(img_points, cv2.MORPH_OPEN, kernel)

# 17.4 闭运算: cv2.morphologyEx() 先进行􏰱􏰲膨胀，􏰷再进行腐蚀，就叫做闭运算
# 用来填充前景物体中的小洞(小􏱂黑点)
# 去黑
img_flaws = cv2.imread('flaws.png')
closing = cv2.morphologyEx(img_flaws, cv2.MORPH_CLOSE, kernel)


# 17.5 形态学梯度: 一幅图像膨胀与腐蚀的差(膨胀 - 腐蚀)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# 17.6 礼帽: 原图 - 开运算图
tophat = cv2.morphologyEx(img_points, cv2.MORPH_TOPHAT, kernel)

# 17.7 黑帽: 原图 - 闭运算图
blackhat = cv2.morphologyEx(img_flaws, cv2.MORPH_BLACKHAT, kernel)

# 17.8 结构化元素: cv2.getStructuringElement(形状, 大小)
# MORPH_RECT        长方形
# MORPH_ELLIPSE     椭圆
# MORPH_CROSS       十字

# 画图打印
images = [img, erosion, dilation, gradient, img_points, opening, tophat, img_flaws, closing, blackhat]
titles = ['Original', 'Erosion', 'Dilation', 'Gradient', 'Points', 'Opening', 'Top Hat',
          'Flaws', 'Closing', 'Black Hat']
DrawImage(images, titles, 3, 4, 10)