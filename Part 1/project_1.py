# 创建色板
# 自由画矩形与曲线
import cv2
import numpy as np


def nothing(x):
    pass

drawing = False  # 鼠标按下为True
mode = True  # mode 为 True 绘制矩形，按下'm'绘制曲线
ix, iy = -1, -1


# 回调函数: 必须 5 个参数
def draw_circle(event, x, y, flags, param):
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    color = (b, g, r)

    global ix, iy, drawing, mode
    # 如果按下鼠标左键返回坐标位置
    if event == cv2.EVENT_LBUTTONDBLCLK:
        drawing = True
        ix, iy = x, y
    # 鼠标在移动且左键被持续按下——>进入绘图模式
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing:
            if mode:    # 绘制矩形
                cv2.rectangle(img, (ix, iy), (x, y), color, -1)
            else:  # 绘制圆圈，填充小圆点连在一起就成了曲线
                cv2.circle(img, (x, y), 1, color, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


print('矩形模式')
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)
cv2.setMouseCallback('image', draw_circle)
while 1:
    cv2.imshow('image', img)
    k = cv2.waitKey(1)
    if k == ord('m'):
        mode = not mode
        print('曲线模式')
    elif k == 27:
        print('退出')
        break
cv2.destroyAllWindows()
