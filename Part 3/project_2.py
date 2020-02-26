import cv2


def nothing(x):
    pass

img = cv2.imread('13.png', 0)

cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
cv2.createTrackbar('thresh', 'Threshold', 0, 150, nothing)

while 1:
    thresh = cv2.getTrackbarPos('thresh', 'Threshold')
    ret, thresh_ = cv2.threshold(img, thresh, 255, 0)
    cv2.imshow('Threshold', thresh_)
    k = cv2.waitKey(1)
    if k == 27:
        print(thresh)
        break
