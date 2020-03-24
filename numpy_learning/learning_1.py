import numpy as np

# 1、矩阵的创建
arr = np.array([[1, 7, 3],
                [4, 0, 6],
                [11, 4, 0],
                [6, 12, 30]], dtype=np.int32)
arr_zero = np.zeros((3, 4), dtype=np.int32) # 创建 3 行 4 列的 零矩阵
arr_one = np.ones((4, 3), dtype=np.int32)   # 创建 4 行 3 列的 E 矩阵
arr_range = np.arange(10, 20, 2)            # 创建 10 ~ 19 步数为 2 的数列
arr_reshape = np.arange(12).reshape((3, 4)) # 创建共 12 个数字, 规定为 4 行 3 列
arr_linesapce = np.linspace(1, 10, 5)       # 创建 1 ~ 10 等距离的 5 个元素的数列
arr_random = np.random.random((4, 4))       # 创建 4 行 4 列 0 ~ 1 之间的随机数矩阵

print(arr)
print(arr.dtype)

print(arr_zero)
print(arr_one)
print(arr_range)
print(arr_reshape)
print(arr_linesapce)
print(arr_random)
print(np.sum(arr_random, axis=1))   # 求每一【行】所有元素的和
print(np.min(arr_random, axis=0))   # 求每一【列】所有元素的最小值
print(np.max(arr_random, axis=1))   # 求每一【行】所有元素的最大值

print("number of dimension:", arr.ndim) # 维度
print("shape:", arr.shape)  # 几行几列
print("size:", arr.size)    # 一共几个数

# 2、矩阵运算
print("============= 简单运算 =============")
print('① arr + arr_one:\n', arr + arr_one)
print('② arr - one:\n', arr - arr_one)
print('③ arr^2:\n', arr ** 2)
print('④ sin(arr):\n', np.sin(arr))
print('⑤ arr > 4\n', arr > 4)

print("============= 矩阵运算 =============")
print('① reshape * arr 1:\n', np.dot(arr_reshape, arr))
print('② reshape * arr 2:\n', arr_reshape.dot(arr))

print("============= 数字特征 =============")
print('arr=\n', arr)
print('① arr 的列平均值:\n', np.mean(arr, axis=0))
print('② arr 的中位数:\n', np.median(arr))
print('③ arr 的累加数列:\n', np.cumsum(arr))
print('④ arr 每行元素之差:\n', np.diff(arr))
print('⑤ arr 非零元素的行与列:\n', np.nonzero(arr))
print('⑥ arr 逐行排序(由大到小):\n', np.sort(arr))
print('⑦ arr 矩阵的转置(np.transpose法):\n', np.transpose(arr))
print('⑧ arr 矩阵的转置(.T法):\n', arr.T)
# arr 中小于 4 的元素变为 4, 大于 10 的元素变成 10
print('⑨ arr 的最大最小值规定:\n', np.clip(arr, 4, 10))

'''
【注意】
axis = 0 -----> 列
axis = 1 -----> 行
'''