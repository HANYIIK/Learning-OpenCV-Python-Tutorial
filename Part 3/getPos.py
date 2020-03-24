"""
点击哪里, 得到哪里的坐标
"""
import cv2

ix, iy = -1, -1


def getPos(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        ix, iy = x, y
        print('[', ix, iy, ']\n')

img = cv2.imread('lane.jpg')
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', getPos)
while 1:
    cv2.imshow('image', img)
    k = cv2.waitKey(1)
    if k == 27:
        print('退出')
        cv2.destroyAllWindows()
        break