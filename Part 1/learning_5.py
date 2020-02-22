# OpenCV中的绘图函数
# 所有绘图函数没有返回值
import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)   # create a black image

# 画线
cv2.line(img, (0, 0), (511, 511), (100, 100, 0), 3)   # Draw a diagonal blue line with thickness of 5 px

# 画矩形
cv2.rectangle(img, (384, 0), (510, 128), (100, 255, 0), 3)

# 画圆
cv2.circle(img, (447, 63), 61, (0, 0, 255), -1) # -1 表示填充

# 画椭圆
cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)

# 画多边形
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))

# 在图片上添加文字
font = cv2.FONT_HERSHEY_SIMPLEX    # 设定字体
cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2)

# 显示
window_name = 'example'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, img)
cv2.waitKey(0)
cv2.destroyAllWindows()