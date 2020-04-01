# 机器学习
# Chapter 2 使用 kNN 对手写数字 OCR
import cv2
import numpy as np


# 显示图像函数
def ShowImage(name_of_image, image_, rate):
    img_min = cv2.resize(image_, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow(name_of_image, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_image, img_min)
    if cv2.waitKey(0) == 27:  # wait for ESC to exit
        print('Not saved!')
        cv2.destroyAllWindows()
    elif cv2.waitKey(0) == ord('s'):  # wait for 's' to save and exit
        cv2.imwrite(name_of_image + '.jpg', image_)  # save
        print('Saved successfully!')
        cv2.destroyAllWindows()

img = cv2.imread('digits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1、图像分割
images = []     # List 型
row_arr = np.vsplit(gray, 50)
for item in row_arr:
    colms_arr = np.hsplit(item, 100)
    # for i in colms_arr:
    #     images.append(i)
    images.append(colms_arr)

# 将 List 型转换为 Array 型
images = np.array(images)

# 2、训练集与测试集选取
TRAIN_DATA = 99
train = images[:, :TRAIN_DATA].reshape(-1, 400).astype(np.float32)
# images[:, :TRAIN_DATA] 是 4 维数组

test = images[:, TRAIN_DATA:100].reshape(-1, 400).astype(np.float32)

img_test = cv2.resize(cv2.imread('other.png', 0), (20, 20), interpolation=cv2.INTER_CUBIC)
test_data = img_test.reshape(-1, 400).astype(np.float32)

'''
① Train Size:
20 x 50 = 1000 行   ——> 4950 行
20 x 99 = 1980 列   ——>  400 列（reshape(-1, 400)）

Train Labels：
[[0]
[0]
…
[0] (495 行)
[1]
[1]
…
[1] (495 行)
.
.
.
[9]
[9]
…
[9] (495 行)]

② Test Size:
20 x 50 = 1000 行   ——>   50 行
20 x 1 = 20 列      ——>  400 列（reshape(-1, 400)）

③ Test Data Size:
20 x 1 = 20 行      ——>    1 行
20 x 1 = 20 列      ——>  400 列（reshape(-1, 400)）
'''
k = np.arange(10)
train_labels = np.repeat(k, TRAIN_DATA * 5)[:, np.newaxis]
# test_labels = np.repeat(k, 500 - TRAIN_DATA * 5)[:, np.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(test_data, k=5)

# matches = result == test_labels
# print('matches:\n', matches)

# correct = np.count_nonzero(matches)
# print('correct:\n', correct)

# accuracy = correct*100.0/result.size
# print('正确率:\n', accuracy, '%')

print(result)