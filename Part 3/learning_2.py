# Chapter 14 几何变换
import cv2
import numpy as np
from matplotlib import pyplot as plt


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

# 14.1 扩展缩放
img = cv2.imread('../Part 2/1.jpg')
res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

ShowImage('img', img, 0.1)
ShowImage('res', res, 0.1)

# 14.2 图像平移
'''
???
'''

# 14.3 图像旋转: getRotationMatrix2D()
rows, cols, channels = img.shape

# 􏰕􏰖第一个参数为旋􏰀中心􏰒, 第二个为旋􏰀􏰁度, 􏰒第三个为旋􏰀后的缩放因子
M = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)

# warpAffine(): 第三个参数是􏰟输出图像的尺寸
dst = cv2.warpAffine(img, M, (2*cols, 2*rows))

ShowImage('dst', dst, 0.1)

# 14.4 仿射变换: getAffineTransform(原坐标集合，现在坐标集合), 搭配 warpAffine()使用
pts1 = np.float32([[500, 500], [2000, 500], [500, 2000]])
pts2 = np.float32([[100, 1000], [2000, 500], [1000, 2500]])

N = cv2.getAffineTransform(pts1, pts2)

dst2 = cv2.warpAffine(img, N, (cols, rows))

# 14.5 透视变换: getPerspectiveTransform(原坐标集合，现在坐标集合), 搭配warpPerspective()使用
'''
???
'''
# 画图
plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst2), plt.title('Output')
plt.show()
