# Chapter 24 霍夫变换
"""
用途：用 霍夫线变换 和 圆变换 可以识别出图像中的 直线 和 圆
函数：cv2.HoughLines(), cv2.HoughLinesP(), cv2.HoughCircles()
"""
import cv2
import numpy as np


# 显示图像函数
def ShowImage(name_of_image, image, rate):
    img_min = cv2.resize(image, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow(name_of_image, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_image, img_min)
    if cv2.waitKey(0) == 27:  # wait for ESC to exit
        print('Not saved!')
        cv2.destroyAllWindows()
    elif cv2.waitKey(0) == ord('s'):  # wait for 's' to save and exit
        cv2.imwrite(name_of_image + '.jpg', image)  # save
        print('Saved successfully!')
        cv2.destroyAllWindows()

img = cv2.imread('hough.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
drawing = np.zeros(img.shape, dtype=np.uint8)
edges = cv2.Canny(img_gray, 50, 150)
# ShowImage('contours', edges, 1)

# 1、标准霍夫直线变换
# 计算图像中的每一个点，计算量比较大
lines = cv2.HoughLines(edges, 0.8, np.pi / 180, 90)
'''
cv2.HoughLines(
① 要检测的二值图（一般是阈值分割或边缘检测后的图）
② 距离 r 的精度，值越大，考虑越多的线
③ 角度θ的精度，值越小，考虑越多的线
④ 累加数阈值，值越小，考虑越多的线
)
'''

length = 1000
# 将检测的线画出来（极坐标）
for line in lines:
    rho, theta = line[0]
    cos = np.cos(theta)
    sin = np.sin(theta)
    x0 = cos * rho
    y0 = sin * rho
    x1 = int(x0 + length * (-sin))
    y1 = int(y0 + length * cos)
    x2 = int(x0 - length * (-sin))
    y2 = int(y0 - length * cos)

    cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 255))

# 2、统计概率霍夫直线变换
lines = cv2.HoughLinesP(edges, 0.8, np.pi / 180, 90,
                        minLineLength=50, maxLineGap=10)
'''
minLineLength：最短长度阈值，比这个长度短的线会被排除
maxLineGap：同一直线两点之间的最大距离
'''
for line in lines:
    x1, y1, x2, y2 = line[0]

    cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1,
             lineType=cv2.LINE_AA)

# ShowImage('hough', drawing, 1)

# 3、霍夫圆变换
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param2=30)
circles = np.int0(np.around(circles))
'''
cv2.HoughCircles(
① 要检测的二值图（一般是阈值分割或边缘检测后的图)
② 变换方法，一般使用霍夫梯度法
③ dp=1：表示霍夫梯度法中累加器图像的分辨率与原图一致
④ 两个不同圆圆心的最短距离
⑤ param2跟霍夫直线变换中的累加数阈值一样
)
'''
for i in circles[0, :]:
    cv2.circle(drawing, (i[0], i[1]), i[2], (255, 0, 0), 2)  # 画出外圆
    cv2.circle(drawing, (i[0], i[1]), 2, (0, 0, 255), 3)  # 画出圆心

ShowImage('hough', drawing, 1)