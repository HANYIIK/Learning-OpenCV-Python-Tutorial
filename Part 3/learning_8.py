# Chapter 20 图像金字塔
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
def DrawPictures(images_set, titles_set, rows, cols, num):
    for i in range(num):
        plt.subplot(rows, cols, i+1), plt.imshow(images_set[i]), plt.title(titles_set[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

