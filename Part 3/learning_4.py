# Chapter 16 图像平滑
"""
LPF 低􏰪通滤波：去除噪音，模糊图像
HPF 􏰞􏰪高通滤波：找到图像的􏰭边缘
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
def ShowImage(name_of_image, image, rate):
    img_min = cv2.resize(image, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow(name_of_image, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_image, img_min)
    SaveOrNot(image)

# 图像模糊技术
# 16.1 平均: cv2.blur()与cv2.boxFilter()
img = cv2.imread('zs.jpg')
blur = cv2.blur(img, (5, 5))

# 16.2 高斯模糊: cv2.GaussianBlur() 􏰇高斯核的宽和􏰇􏰀高必􏰢是[奇数] odd􏰁
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# 16.3 中值模糊: cv2.medianBlur() 与卷积框对应像素的中值来替代中心像素的值, 用于去􏰅椒盐噪声(斑点)
median = cv2.medianBlur(img, 5)
# 16.4 双边滤波: cv2.bilateralFilter() 在【保持􏰋边界】清晰的情况下有效的去去除噪音
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

plt.subplot(231), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.imshow(blur), plt.title('Blur')
plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.imshow(gaussian), plt.title('Gaussian Blur')
plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(median), plt.title('Median Blur')
plt.xticks([]), plt.yticks([])

plt.subplot(235), plt.imshow(bilateral), plt.title('Bilateral Blur')
plt.xticks([]), plt.yticks([])

plt.show()

'''
【小结】
在不知道用什么滤波器好的时候，优先高斯滤波cv2.GaussianBlur()，然后均值滤波cv2.blur()。
斑点和椒盐噪声优先使用中值滤波cv2.medianBlur()。
要去除噪点的同时尽可能保留更多的边缘信息，使用双边滤波cv2.bilateralFilter()。

【线性】滤波方式：均值滤波、方框滤波、高斯滤波（速度相对快）
【非线性】滤波方式：中值滤波、双边滤波（速度相对慢)
'''