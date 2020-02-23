# Chapter 18 图像梯度
"""
三种梯度滤波器
"""
import cv2
import numpy as np
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


# 显示图集函数
def ShowPictures(images_set_2, titles_set_2, num_2, rate_2):
    for i in range(num_2):
        ShowImage(titles_set_2[i], images_set_2[i], rate_2)


# 画图集函数
def DrawImage(images_set, titles_set, rows, cols, num):
    for i in range(num):
        plt.subplot(rows, cols, i+1), plt.imshow(images_set[i]), plt.title(titles_set[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


# 18.1 Sobel 算子和 Scharr 算子: 用来求一阶或二阶􏱆导数 Scharr 是对 Sobel􏰀（使用小的卷积核求解􏰎梯度􏱇角度时）􏰁的优化
# 18.2 Laplacian 算子: 用来求二阶􏱆导数
'''
Sobel 算子：高􏰄斯平滑与微分操作的结合体，􏰀抗噪声能力很好
Scharr 算子：如果 ksize = -1􏰀 会使用 3x3 的 Scharr 滤波器􏰀，效果􏰍 比 3x3 的 Sobel 滤波器好
在使用 3x3 滤波器时应􏰏尽量使用 Scharr 滤波器
'''
img = cv2.imread('3.jpg', 0)

# cv2.CV_64F 输出图像的深度（数据类型），可以使用 -1， 与原图像保持一致 np.uint8
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# 求 x 方向求一阶导数
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

# 求 y 方向求一阶导数
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)


# 不同深度对输出图像的影响: CV_64F 与 CV_8U
img2 = cv2.imread('zs.jpg', 0)

# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img2, cv2.CV_8U, 1, 0, ksize=5)

# Output dtype = cv2.CV_64F
sobelx64f = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=5)

# Then take its absolute
abs_sobel64f = np.absolute(sobelx64f)

# and convert to cv2.CV_8U
sobel_8u = np.uint8(abs_sobel64f)

images = [img2, sobelx8u, sobel_8u]
titles = ['Original', 'Sobel CV_8U', 'Sobel abs(CV_64F)']
ShowPictures(images, titles, 3, 1)