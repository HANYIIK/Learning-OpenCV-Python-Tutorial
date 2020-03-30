# 特征提取
# Chapter 2 Shi-Tomasi角点检测算法
"""
函数: cv2.goodFeatureToTrack()
函数用途: 可以获取图像中 N 个最好的􏰀角点
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

img = cv2.imread('chessboard.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi角点检测
'''
cv2.goodFeatureToTrack(
① 灰度图 - img
② 
③ 
④ 
)
'''
corners = cv2.goodFeaturesToTrack(img_gray, 20, 0.01, 10)
corners = np.int0(corners)
# print(len(corners))
# 12个角点坐标

for i in corners:
    # 压缩至一维：[[62,64]]->[62,64]
    # x, y = i.ravel()
    cv2.circle(img, tuple(i.ravel()), 4, (0, 0, 255), -1)

# ShowImage('res', img, 1)
