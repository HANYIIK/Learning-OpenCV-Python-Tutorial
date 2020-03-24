# Chapter 23 模式匹配
"""
① 定义：在大图中找小图，在一副图像中寻找另外一张模板图像的位置
② 用途：可以使用模板匹配在图像中寻找物体
③ 函数：cv2.matchTemplate(), cv2.minMaxLoc()
④ cv2.matchTemplate()返回值：一副灰度图，最白的地方表示最大的匹配
⑤ cv2.minMaxLoc()返回值：最大匹配值的坐标
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


img_bgr = cv2.imread('full.png')
img_compare = img_bgr.copy()
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

img_part = cv2.imread('coin.png', 0)
h, w = img_part.shape   # rows --> h, cols --> w

# ① 相关系数匹配法：cv2.TM_CCOEFF 匹配一个
# res = cv2.matchTemplate(img, img_part, cv2.TM_CCOEFF)

# ② 标准相关模板匹配：cv2.TM_CCOEFF_NORMED 匹配全部
res = cv2.matchTemplate(img, img_part, cv2.TM_CCOEFF_NORMED)
ShowImage('res', res, 1)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# left_top = max_loc  # 左上角
# right_bottom = (left_top[0] + w, left_top[1] + h)   # 右下角
# cv2.rectangle(img_bgr, left_top, right_bottom, (0, 0, 255), 2)  # 画出矩形位置

threshold = 0.8
loc = np.where(res >= threshold)
# 返回的坐标 loc 是[([z],) [y], [x]]格式的，可以用loc[::-1]反转一下, 变成[[x], [y], ([z])]
for lt in zip(*loc[::-1]):
    rb = (lt[0] + w, lt[1] + h)
    cv2.rectangle(img_bgr, lt, rb, (0, 0, 255), 2)
ShowImage('mario', np.hstack((img_compare, img_bgr)), 1)