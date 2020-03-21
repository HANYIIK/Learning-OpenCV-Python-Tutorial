"""
自制阈值定位小工具
"""
import cv2


def nothing(x):
    pass

# img = cv2.imread('epsilon.png', 0)
img = cv2.imread('logo_gray.jpg', 0)

cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
cv2.createTrackbar('thresh', 'Threshold', 0, 255, nothing)

while 1:
    thresh = cv2.getTrackbarPos('thresh', 'Threshold')
    # 色值高于 thresh 的一律变成填充色白色（255)
    ret, thresh_ = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    cv2.imshow('Threshold', thresh_)
    k = cv2.waitKey(1)
    if k == 27:
        print('thresh = ', thresh)
        break
