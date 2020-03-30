# 特征提取
# Chapter 1 Harris 角点检测算法
"""
函数: cv2.cornerHarris(), cv2.cornerSubPix()
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

# 1.1 Harris 角点检测(粗略版)
'''
cv2.cornerHarris(
① 数据类型为 float32 的􏱃输入图像 - img, 
② 􏰎角点检测中􏰀要考虑􏱄的领域大小 - blockSize, 
③ Sobel 求导中使用的窗口大小 - ksize, 
④ Harris 角􏰎点检测方程中的自由参数􏰄，取值参数为 [0,04, 􏰄0.06] - k
)
'''
img = cv2.imread('chessboard.png')

# 1、Harris 角点检测基于【灰度】图像
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# type of img/img_gray is 【<class 'numpy.ndarray'>】(np.uint8)

# 2、Harris 角点检测
dst = cv2.cornerHarris(img_gray, 2, 3, 0.04)

# 3、腐蚀一下，便于标记
dst = cv2.dilate(dst, None)

# 4、角点标记为红色
# img[dst > 0.01 * dst.max()] = [0, 0, 255]
# ShowImage('test', img, 10)

# 1.2 Harris 角点检测(精准版)
# 亚像素级精确度的􏰎角点（小角点）
'''
cv2.cornerSubPix(
① 灰度图 - img
② 角点 - corners
③ winSize
④ zeroZone
⑤ 标准 - criteria
)
'''
ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
dst = np.uint8(dst)

# 找到形心 centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# 定义一个提取角点的标准
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

corners = cv2.cornerSubPix(img_gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

print('criteria(标准) = \n', criteria)
print('centroids(形心) = \n', centroids)
print('corners(角点) = \n', corners)

res = np.hstack((centroids, corners))

print('before int0, res(形心, 角点) 小数版 = \n', res)

# 将 res 的所有元素取整(非四舍五入)
res = np.int0(res)

print('after int0, res(形心, 角点) 整数版 = \n', res)

img[res[:, 1], res[:, 0]] = [0, 0, 255]
img[res[:, 3], res[:, 2]] = [0, 255, 0]

# ShowImage('test', img, 2)
