# 图片读取与保存
import cv2


def SaveOrNot(k, image):
    if k == 27:  # wait for ESC to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):  # wait for 's' to save and exit
        cv2.imwrite('test.png', image)  # save
        cv2.destroyAllWindows()


img = cv2.imread('1.jpg', cv2.IMREAD_COLOR)  # read pic
# cv2.IMREAD_COLOR: 彩色
# IMREAD_GRAYSCALE: 黑白灰

cv2.namedWindow('RUGU', cv2.WINDOW_NORMAL)   # cv2.WINDOW_NORMAL：可调整窗口大小

cv2.imshow('RUGU', img)
# 绑定键盘，等待键盘输入
m = cv2.waitKey(0)
SaveOrNot(m, img)
