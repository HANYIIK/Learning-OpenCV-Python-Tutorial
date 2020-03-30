"""
numpy 的赋值
"""
import numpy as np

A = np.arange(3, 15).reshape((3, 4))
B = np.arange(10, 22).reshape((3, 4))
print('A =\n', A)
print('B=\n', B)
print('A >= 6:\n', A >= 6)          # Boolean 型 3x4 数组
print('B[A >= 6]:\n', B[A >= 6])    # 一个一维数组, 由 B 在 A >= 6 相同位置的数构成

B[A >= 6] = 0                       # 给 B 在 A >= 6 相同位置的数赋值
print('B new =\n', B)