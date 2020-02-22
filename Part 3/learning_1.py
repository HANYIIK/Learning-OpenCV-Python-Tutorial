# Chapter 13 颜色空间转换
import cv2
from numpy.core.defchararray import startswith
import numpy as np

# 13.1 转换颜色空间
flag = 0
for i in dir(cv2):
    if startswith(i, 'COLOR_'):
        flag += 1
print(flag)     # 共有 274 种 cv2.COLOR_xxx

# 13.2 物体跟踪


def FindBlue():   # 识别视频中的蓝色物体并跟踪显示
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # 获取每一帧
        ret, frame = cap.read()
        # 转换到 HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 设定蓝色的阈值
        lower_blue = np.array([100, 43, 46])
        upper_blue = np.array([124, 255, 255])
        # 根据阈值构建 mask
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # 对原图像和 mask 进行位运算
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # 显示图片
        cv2.imshow('src', res)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def BGR2HSV(a):     # 找到 BGR 对应的 HSV 值
    return cv2.cvtColor(np.uint8([[a]]), cv2.COLOR_BGR2HSV)[0][0]


def HSV2BGR(b):     # 找到 HSV 对应的 BGR 值
    return cv2.cvtColor(np.uint8([[b]]), cv2.COLOR_HSV2BGR)[0][0]
