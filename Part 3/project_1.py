"""
自制 Canny 边缘检测小工具
"""
import cv2


def nothing(x):
    pass

img = cv2.imread('logo_gray.jpg', 0)

cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Min', 'edges', 0, 100, nothing)
cv2.createTrackbar('Max', 'edges', 0, 100, nothing)

while 1:
    min_bar = cv2.getTrackbarPos('Min', 'edges')
    max_bar = cv2.getTrackbarPos('Max', 'edges')
    edges = cv2.Canny(img, min_bar, max_bar)
    cv2.imshow('edges', edges)
    k = cv2.waitKey(1)
    if k == 27:
        print('Min: ', min_bar)
        print('Max: ', max_bar)
        break

'''
the best: min = 54 / max = 82
'''