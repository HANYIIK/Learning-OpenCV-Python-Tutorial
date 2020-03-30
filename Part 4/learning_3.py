# 特征提取
# Chapter 3 SIFT 角点检测算法
"""
原理: 尺度不变特征变换。这个算法图片不管放大还是缩小都能找到角点, 可以提取图像中的关键点并􏰷计算它们的描述符。
特点: 多尺度描述信息，能够有效描述缩放，并且对 图像旋转、亮度、仿射变换、视角变化具有很好的适应性
函数:
"""
import cv2


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

img = cv2.imread('1.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# type(sift)    ----> <class 'cv2.xfeatures2d_SIFT'>

# detect()可以在图像中找到关键点
kp = sift.detect(img_gray, None)
# type(kp)      ----> list
# type(kp[1])   ----> <class 'cv2.KeyPoint'>

cv2.drawKeypoints(img_gray, kp, img,
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
ShowImage('test', img, 0.5)