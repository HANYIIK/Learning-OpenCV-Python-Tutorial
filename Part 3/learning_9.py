# Chapter 21 轮廓
"""
找轮廓，绘制轮廓
cv2.findContours()􏰂
cv2.drawContours()
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

# 【21.1 在一幅图像上绘制所有轮廓】
# ① 以灰度图读取原图
img = cv2.imread('13.png', 0)

# ② 阈值化图像: 【轮廓是白色的部分】
# ret, thresh = cv2.threshold(img, 61, 255, 0)
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ③ 找到轮廓
# image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image, contours, hierarchy = cv2.findContours(thresh, 3, 2)

# 转换颜色为彩色图(将轮廓画于此)
img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_copy = img_bgr.copy()

# ④ 绘制独立轮廓
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

# 【21.2.2 轮廓面积 cv2.contourArea()】
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
# 找凸缺陷时一定要设置为 False !
hull = cv2.convexHull(cnt)

# 21.2.6 凸性检测: cv2.isContourConvex()
# 可以可以用来检测一个曲线是不是凸的
# 只返􏰕回 True 或 False
print('凸性检测结果:', cv2.isContourConvex(cnt))

# 21.2.7 边界矩形
# ① 直边界矩形: 不会考虑对象是否旋转􏰪，外接矩形
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

# ② 旋转的边界矩形: 会考虑对象是否旋转􏰪，最小外接矩形
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

# ShowImage('外接矩形与外接圆', img_bgr, 1)

'''
【小结】
cv2.contourArea()算面积
cv2.arcLength()算周长
cv2.boundingRect()算外接矩
cv2.minAreaRect()算最小外接矩
cv2.minEnclosingCircle()算最小外接圆
cv2.matchShapes()进行形状匹配(project_3)
'''

# 【21.3 轮廓性质】
# 21.3.1 宽高比
aspect_ratio = float(w)/h
print('宽高比 = ', aspect_ratio)

# 21.3.2 Extent: 轮廓面积与边界矩形面积的比
rect_area = w * h
extent = float(area)/rect_area
print('Extent = ', extent)

# 21.3.3 Solidity: 轮廓面积与凸包面积的比
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area
print('Solidity = ', solidity)

# 21.3.4 Equivalent Diameter: 与轮廓面积相等的圆形直径
equi_diameter = np.sqrt(4*area/np.pi)
print('Equivalent Diameter = ', equi_diameter)

# 21.3.5 方向
(x_, y_), (MA, ma), angle = cv2.fitEllipse(cnt)
print('x = ', x_, '\ny = ', y_, '\nMA = ', MA, '\nma = ', ma, '\nangle = ', angle)

# 21.3.6 掩模和像素点
mask = np.zeros(img.shape, np.uint8)
cv2.drawContours(mask, [cnt], 0, 255, -1)
pixelpoints = np.transpose(np.nonzero(mask))
# pixelpoints = cv2.findNonZero(mask)
print('像素点:\n', pixelpoints)

# 21.3.7 最大值与最小值以及它们的位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img, mask=mask)
print('min_val = ', min_val, '\nmax_val = ', max_val, '\nmin_loc = ',
      min_loc, '\nmax_loc = ', max_loc)

# 21.3.8 平均颜色以及平均灰度
mean_val = cv2.mean(img_bgr, mask=mask)
print('平均颜色:', mean_val)

# 21.3.9 极点: 一个对象最上面、最下面、最左边、最右边的点
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
print('左极点:', leftmost, '\n右极点:', rightmost,
      '\n上极点:', topmost, '\n下极点:', bottommost)

# 【21.4 轮廓的更多函数】
# 21.4.1 凸缺陷
hull = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)
for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img_copy, start, end, [0, 255, 0], 2)   # green
    cv2.circle(img_copy, far, 5, [0, 0, 255], -1)
# ShowImage('凸缺陷', img_copy, 1)

# 21.4.2 Point Polygon Test: 图像中的一个点到轮􏰃􏰀廓的最短􏰠距离
# 如果点在􏰀轮廓的外􏰾部，返􏰔回值为 负数􏱰；
# 如果在􏰀轮廓上，返􏰒􏰔回值为 0；
# 如果在轮􏰀廓内部，返􏰾􏰒􏰔回值为 正数。
dist = cv2.pointPolygonTest(cnt, (150, 150), True)
# 只有第三个参数设置为 True 时，回计算最短距离
# 否则返回该点与轮廓的位置关系（-1 or 1 or 0)
# print(dist)

# 21.4.3 形状匹配
# (见 project_3)

# 【21.5 轮廓的层次结构】
hierarchy_img = cv2.imread('hierarchy.jpg')

img_hierarchy_gray = cv2.cvtColor(hierarchy_img, cv2.COLOR_BGR2GRAY)

hierarchy_ret, hierarchy_thresh = cv2.threshold(img_hierarchy_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

hierarchy_image, hierarchy_contours, hierarchy_hierarchy = cv2.findContours(hierarchy_thresh, cv2.RETR_TREE, 2)

# print(len(hierarchy_contours), hierarchy_hierarchy)
# 8 条
cv2.drawContours(hierarchy_img, hierarchy_contours, -1, (0, 0, 255), 2)
# ShowImage('hierarchy', hierarchy_img, 1)

'''
轮廓检索模式
cv2.findContours(输入图像, 检索模式, 近似方法)
① RETR_LIST     ---> 不建立轮廓间的子属关系，所有轮廓都属于同一层级
② RETR_TREE     ---> 会完整建立轮廓的层级从属关系
③ RETR_EXTERNAL ---> 只寻找最高层级的轮廓
④ RETR_CCOMP    ---> 把所有的轮廓只分为2个层级，不是外层的就是里层的
'''
