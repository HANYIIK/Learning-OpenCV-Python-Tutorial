# Chapter 20 图像金字塔与图像融合
"""
【同一图像】【不同分辨率】的子图集合
分辨率最【大】的图像放在【底】部, 分辨率最【小】的放在【􏰐顶】部􏰑􏰐
分类：
􏰓① 高斯金字塔: 顶部是通过􏰑􏰐􏰔􏰕将底部􏰐图像中的􏰖连续的􏰉行和列去除􏰗得到的
(我的理解：每次计算，长/2，宽/2，尺寸越来越小，由【高分辨率】大图生【低分辨率】小图)
② 拉普拉斯金字塔: 顶部是通过􏰑􏰐􏰔􏰕将底部􏰐图像中的􏰖连续的􏰉行和列乘积􏰗得到的
(我的理解：每次计算，长*2，宽*2，尺寸越来越大，由【低分辨率】小图生【低分辨率】大图)
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

apple_1 = cv2.imread('apple.png')
apple = cv2.resize(apple_1, (256, 256))
orange_1 = cv2.imread('Orange.jpg')
orange = cv2.resize(orange_1, (256, 256))

# 生成 apple 的高斯金字塔
gaussian = apple.copy()
group_gaussian_apple = [gaussian]
for i in range(6):
    gaussian = cv2.pyrDown(gaussian)
    group_gaussian_apple.append(gaussian)

# 生成 orange 的高斯金字塔
gaussian = orange.copy()
group_gaussian_orange = [gaussian]
for i in range(6):
    gaussian = cv2.pyrDown(gaussian)
    group_gaussian_orange.append(gaussian)

# 生成 apple 的拉普拉斯金字塔
group_laplacian_apple = [group_gaussian_apple[5]]
for i in range(5, 0, -1):
    laplacian = cv2.pyrUp(group_gaussian_apple[i])
    group_laplacian_apple.append(laplacian)

# 生成 orange 的拉普拉斯金字塔
group_laplacian_orange = [group_gaussian_orange[5]]
for i in range(5, 0, -1):
    laplacian = cv2.pyrUp(group_gaussian_orange[i])
    group_laplacian_orange.append(laplacian)

LS = []
for la, lo in zip(group_laplacian_apple, group_laplacian_orange):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:int(cols / 2)], lo[:, int(cols/2):]))
    LS.append(ls)

ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

ShowImage('test', ls_, 1)