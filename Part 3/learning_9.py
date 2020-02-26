# Chapter 21 轮廓
"""
找轮廓，绘制轮廓
cv2.findContours()􏰂
cv2.drawContours()
"""
import cv2
from matplotlib import pyplot as plt


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
def ShowImage(name_of_image, image_, rate):
    img_min = cv2.resize(image_, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow(name_of_image, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_image, img_min)
    SaveOrNot(image_)


# 显示图集函数
def ShowPictures(images_set_2, titles_set_2, num_2, rate_2):
    for i in range(num_2):
        ShowImage(titles_set_2[i], images_set_2[i], rate_2)


# 画图集函数
def DrawPictures(images_set, titles_set, rows, cols, num):
    for i in range(num):
        plt.subplot(rows, cols, i+1), plt.imshow(images_set[i]), plt.title(titles_set[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

'''
图像, 轮廓, 轮廓的层次结构 = cv2.findContours(输入图像, 检索模式, 近似方法)
􏰀廓􏰇第二个􏰚返回值􏰉是一个 Python 列􏰛􏰃表，其中存储􏰜图像中的所有􏰀轮廓，
每一个􏰀轮廓􏰝是一个 Numpy 数组􏰃，包含对􏰞􏰈像边界点􏰇(x, y)􏰉的坐标。
'''

'''
cv2.drawContours(图像, 
轮廓【一个 Python 列表】, 
轮廓的索引【当设置为 -1 时绘制所有轮廓】) 
可以􏰢用来绘制轮廓
'''

# 在一幅图像上绘制所有轮廓
# 1、以灰度图读取原图
img = cv2.imread('13.png', 0)

# 2、阈值化图像: 【轮廓是白色的部分】
ret, thresh = cv2.threshold(img, 61, 255, 0)

# 3、找到轮廓
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 4、绘制独立轮廓
line = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

ShowImage('Contour', line, 1)
