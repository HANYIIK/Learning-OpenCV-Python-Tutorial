# Chapter 10 图像上的算术运算
import cv2
import numpy as np

e1 = cv2.getTickCount()


# 保存图像函数
def SaveOrNot(pic):
    if cv2.waitKey(0) == 27:  # wait for ESC to exit
        print('Not saved!')
        cv2.destroyAllWindows()
    elif cv2.waitKey(0) == ord('s'):  # wait for 's' to save and exit
        cv2.imwrite('result.jpg', pic)  # save
        print('Saved successfully!')
        cv2.destroyAllWindows()


# 显示图像函数
def ShowImage(name_of_image, image, rate):
    data = image.shape
    img = cv2.resize(image, (int(data[0]/rate), int(data[1]/rate)), interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow(name_of_image, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_image, img)
    SaveOrNot(img)

# 10.1 图像加法`
'''
x = np.uint8([250])
y = np.uint8([10])
# OpenCV的加法
print(cv2.add(x, y))    # 250 + 10 = 260 ===> 255
# Numpy的加法
print(x+y)              # 250 + 10 = 260 % 256 = 4
'''


# 10.2 图像混合


# 混合两张图片函数
def MixImage(pic1, degree1, pic2, degree2):
    # addWeighted() 混合函数要求两张图片大小要一样
    if pic1.size == pic2.size:
        dst = cv2.addWeighted(pic1, degree1, pic2, degree2, 0)
        cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
        cv2.imshow('dst', dst)
        SaveOrNot(cv2.waitKey(0))
    else:
        print('Size not match!')

'''
img1 = cv2.imread('test_1.jpg')
img2 = cv2.imread('test_2.jpg')
MixImage(img1, 0.5, img2, 0.5)
'''

# 10.3 按位运算：cv2.bitwise_and/or/not/xor
# 将 OpenCV 的 logo 放在背景图的左上角：

# 读入图片
logo = cv2.imread('logo.jpg')   # logo
me = cv2.imread('me.jpg')       # me

# 创造一个背景 me 的 ROI 用来放 logo
rows, cols, channels = logo.shape
me_roi = me[0:rows, 0:cols]
ShowImage('test', me_roi, 4)

# ==================== 核心操作 ====================
# 取 logo 的单通道灰色模式
logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ShowImage('logo_gray', logo_gray, 4)

# 色值高于 20 的一律变成填充色白色（255）。黑色=0; 白色=255
ret, mask = cv2.threshold(logo_gray, 20, 255, cv2.THRESH_BINARY)
ShowImage('mask', mask, 4)

# 取反，黑色变白色，白色变黑色
mask_inv = cv2.bitwise_not(mask)
ShowImage('mask_inv', mask_inv, 4)

# 将 ROI 中 mask_inv 为0的部分（黑色部分）置0（变黑），即把logo挖空
me_bg = cv2.bitwise_and(me_roi, me_roi, mask=mask_inv)
ShowImage('me_bg', me_bg, 4)

# 将彩色 logo 中的 mask 为0的部分（黑色部分）置0（变黑），即把背景挖空
me_fg = cv2.bitwise_and(logo, logo, mask=mask)
ShowImage('me_fg', me_fg, 4)

# 把彩色的 logo 放在背景上
dst_1 = cv2.add(me_bg, me_fg)
ShowImage('dst_1', dst_1, 4)

# 2D卷积: cv2.filter2D()模糊
kernel = np.ones((5, 5), np.float32)/25
dst_2 = cv2.filter2D(dst_1, -1, kernel)     # 卷积操作，-1表示通道数与原图相同
ShowImage('dst_2', dst_2, 4)

# 替换原图中的部分，大功告成
me[0:rows, 0:cols] = dst_2

# 显示成果
ShowImage('res', me, 4)

# 计算执行时间
e2 = cv2.getTickCount()
time = (e2 - e1)/cv2.getTickFrequency()
print('TIME:' + str(time))
