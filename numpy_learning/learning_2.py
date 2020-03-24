import numpy as np

"""
3、Array 迭代
"""
A = np.arange(3, 15).reshape((3, 4))
print(A)

print(A.flatten())

# 迭代行
for row in A:
    print(row)
# 迭代列
for column in A.T:
    print(column)

"""
4、Array 合并
"""
B = np.array([1, 1, 1])
C = np.array([2, 2, 2])

print(np.vstack((B, C)))    # Vertical Stack ---> 垂直合并
print(np.hstack((B, C)))    # Horizontal Stack ---> 水平合并

print(B[np.newaxis, :])     # 在行上加一个维度
print(B[:, np.newaxis])     # 在列上加一个维度

# 多数列横向合并
print(np.concatenate((B, C, C, B), axis=0))
