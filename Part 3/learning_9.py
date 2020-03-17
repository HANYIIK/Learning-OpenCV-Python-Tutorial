# Chapter 21 轮廓
"""
找轮廓，绘制轮廓
cv2.findContours()􏰂
cv2.drawContours()
"""
import cv2
import numpy as np


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

'''
图像, 轮廓, 轮廓的层次结构 = cv2.findContours(输入图像, 检索模式, 近似方法)
􏰀􏰇第二个􏰚返回值􏰉是一个 Python 列􏰛􏰃表，存储􏰜图像中的所有􏰀轮廓，
每一个􏰀轮廓􏰝是一个 Numpy 数组􏰃，包含对􏰞􏰈像边界点􏰇(x, y)􏰉的坐标。
'''

'''
cv2.drawContours(图像（轮廓会画在这个图像上）, 
轮廓【一个 Python 列表】, 
轮廓的索引【当设置为 -1 时绘制所有轮廓】) 
可以􏰢用来绘制轮廓
'''

# 21.1 在一幅图像上绘制所有轮廓
# 1、以灰度图读取原图
img = cv2.imread('13.png', 0)

# 2、阈值化图像: 【轮廓是白色的部分】
# ret, thresh = cv2.threshold(img, 61, 255, 0)
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3、找到轮廓
# image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image, contours, hierarchy = cv2.findContours(thresh, 3, 2)

# 转换颜色为彩色图(将轮廓画于此)
img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 4、绘制独立轮廓
cnt = contours[0]
# cv2.drawContours(img_bgr, [cnt], 0, (0, 0, 255), 2) ----> 只会绘制"3"的轮廓
cv2.drawContours(img_bgr, contours, -1, (0, 0, 255), 3)

# 21.2 轮廓特征: 面积、周长、重心、边界框 cv2.moments()
# 21.2.1 矩: 计算图像质心
M = cv2.moments(cnt)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
# print(cx)
# print(cy)
# cx = 121
# cy = 110

# 21.2.2 轮廓面积 cv2.contourArea()
area = cv2.contourArea(cnt)
# print(area)

# 21.2.3 轮廓周长
perimeter = cv2.arcLength(cnt, True)
# print(perimeter)

# 21.2.4 轮廓近似: 调整轮廓准确的程度
epsilon = 0.01 * perimeter
approx = cv2.approxPolyDP(cnt, epsilon, True)
# ShowImage('test', img, 1)

# 21.2.5 凸包: cv2.convexHull()
# 用来检测一个曲线是否有凸性缺陷，并纠正缺陷
# 如果一个凸性曲线是凹的，就叫【凸性缺陷】
# hull = cv2.convexHull(points, hull, clockwise, returnPoints)
# points: 传入的轮廓
# hull: 输出，通常不需要
# clockwise: 方向标志。True---输出的凸包是顺时针方向，否则逆时针方向
# returnPoints: 默认值True---返回凸包上点的坐标，否则返回凸包点对应的轮廓上的点
# hull = cv2.convexHull(cnt)

# 21.2.6 凸性检测: cv2.isContourConvex()
# 可以可以用来检测一个曲线是不是凸的
# 只返􏰕回 True 或 False
print(cv2.isContourConvex(cnt))

# 21.2.7 边界矩形
# 直边界矩形: 不会考虑对象是否旋转􏰪，外接矩形
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 旋转的边界矩形: 会考虑对象是否旋转􏰪，最小外接矩形
rect = cv2.minAreaRect(cnt)
box = np.int0(cv2.boxPoints(rect))  # 外接矩形的四个角点取整
cv2.drawContours(img_bgr, [box], 0, (255, 0, 0), 2)

# 21.2.8 最小外接圆
(x, y), radius = cv2.minEnclosingCircle(cnt)
(x, y, radius) = np.int0((x, y, radius))    # 圆心、半径取整
cv2.circle(img_bgr, (x, y), radius, (255, 0, 255), 2)

# 21.2.9 拟合椭圆
ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(img_bgr, ellipse, (0, 255, 255), 2)
ShowImage('外接矩形与外接圆', img_bgr, 1)

'''
cv2.contourArea()算面积
cv2.arcLength()算周长
cv2.boundingRect()算外接矩
cv2.minAreaRect()算最小外接矩
cv2.minEnclosingCircle()算最小外接圆
cv2.matchShapes()进行形状匹配(project_3)
'''