# Chapter 15 图像阈值
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

# 15.1 简单阈值: threshhold(灰度图, 阈值, 高于阈值时设置的新值(0黑 or 255白), 方法)
'''
img = cv2.imread('2.jpg', 0)

ret1, thresh1 = cv2.threshold(img, 127, 200, cv2.THRESH_BINARY)          # 高于阈值的部分置200，其余部分置黑
ret2, thresh2 = cv2.threshold(img, 127, 200, cv2.THRESH_BINARY_INV)      # 低于阈值的部分置200，其余部分置黑
ret3, thresh3 = cv2.threshold(img, 127, 200, cv2.THRESH_TRUNC)           # 高于阈值的部分置阈值的值
ret4, thresh4 = cv2.threshold(img, 127, 200, cv2.THRESH_TOZERO)          # 低于阈值被置黑(第3个参数无效)
ret5, thresh5 = cv2.threshold(img, 127, 200, cv2.THRESH_TOZERO_INV)      # 高于阈值被置黑(第3个参数无效)


titles = ['Original', 'BINARY', 'BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(231+i), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
'''

# 15.2 自适应阈值
'''
# 中值滤波
img = cv2.medianBlur(img, 5)
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# 11 为 Block size, 2 为 C 值
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY, 5, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 5, 2)
titles = ['Original', 'Global Threshold(v = 127)',
          'Adaptive Mean Threshold', 'Adaptive Gaussian Threshold']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(231+i), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
'''

# 15.3 Otsu's二值化
img = cv2.imread('zs.jpg', 0)

# 固定阈值法
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu阈值法
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 先进行高斯滤波，再使用Otsu阈值法
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# plot
titles = ['Original', 'Histogram', 'Global Threshold(v=127)',
          'Original', 'Histogram', "Otsu's Threshold",
          'Gaussian filtered Image', 'Histogram', "Otsu's Threshold"]
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
for i in range(3):
    plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

    plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])

    plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.show()