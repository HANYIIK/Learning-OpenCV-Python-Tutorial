import numpy as np

"""
5、Array 分割
"""
A = np.arange(2, 26).reshape((6, 4))
print('A =\n', A)

# 等量的分割
# split(分割对象矩阵, 分割成几个部分, 分割行(1)还是分割列(0))
print('① 等量的分割:\n', np.split(A, 2, axis=0))

# 不等量的分割
print('② 不等量的分割:\n', np.array_split(A, 4, axis=0))

# 纵向分割 Vertical Split
print('③ 纵向分割 Vertical Split:\n', np.vsplit(A, 3))

# 横向分割 Horizontal Split
print('④ 横向分割 Horizontal Split:\n', np.hsplit(A, 2))

"""
6、Array 赋值
"""
B = A
B[1, 2] = 80

print('赋值后每一行从小到大排序好的 B:\n', np.sort(B))
print('B is A ?', B is A)   # True
'''
【注意：numpy 中 array 的自动关联性质】
改变 A 就是改变 B
改变 B 就是改变 A
【不想关联就用 B = A.copy()】
'''
C = A.copy()
A[1, 2] = 100
print('C is A ?', C is A)   # False
print('A =\n', A)
print('C =\n', C)