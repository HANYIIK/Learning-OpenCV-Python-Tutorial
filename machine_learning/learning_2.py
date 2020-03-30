# 机器学习
# Chapter 2 使用 kNN 对手写数字 OCR
import cv2
import numpy as np

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
TRAIN_DATA = 90
train = images[:, :TRAIN_DATA].reshape(-1, 400).astype(np.float32)
test = images[:, TRAIN_DATA:100].reshape(-1, 400).astype(np.float32)

k = np.arange(10)
train_labels = np.repeat(k, TRAIN_DATA * 5)[:, np.newaxis]
test_labels = np.repeat(k, 500 - TRAIN_DATA * 5)[:, np.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(test, k=5)

matches = result == test_labels
# print('matches:\n', matches)

correct = np.count_nonzero(matches)
# print('correct:\n', correct)

accuracy = correct*100.0/result.size
print('正确率:\n', accuracy, '%')