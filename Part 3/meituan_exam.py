"""
美团机试题
【题目】
两行 n 列矩阵，要求主角从(1, 1)点出发，最终要到达(2, n)点。
主角只有这三种走法:
①向右:   (x, y)---> (x, y+1)
②向右下: (x, y)---> (x+1, y+1)
③向右上: (x, y)---> (x-1, y+1)
遇到'X'表示不能落脚
遇到'.'表示可以落脚
【输入】
第一行输入 n
第二行输入 1 * n 矩阵，如'...X.X.X.X.....'
第三行输入 1 * n 矩阵，如'..X........XXX.'
【输出】
可到达目的地的方法种类个数(不可到达输出 -1)
"""
import numpy as np
import math

global n

n = int(input())
blog = np.ones((2, n), np.str)
str0 = input()
str1 = input()
# 初始化blog
j = 0
for i in str0:
    blog[0][j] = i
    j += 1
j = 0
for i in str1:
    blog[1][j] = i
    j += 1


# 向右移动
def goRight(my_blog, pos):
    if pos[1] == n - 1:
        return pos
    if my_blog[pos[0]][pos[1] + 1] == 'X':
        return pos
    else:
        return [pos[0], pos[1] + 1]


# 向右上或右下移动
def goRightOrLeft(my_blog, pos):
    if pos[1] == n - 1:
        return pos
    # 右上
    if pos[0] == 1:
        if my_blog[0][pos[1] + 1] == 'X':
            return pos
        else:
            return [0, pos[1] + 1]
    # 右下
    elif pos[0] == 0:
        if my_blog[1][pos[1] + 1] == 'X':
            return pos
        else:
            return [1, pos[1] + 1]
    else:
        return pos


position = [0, 0]
flag = 0
methods = int(math.pow(2, n - 1) - 1)
method = 0
method_bin = str('{:050b}'.format(method)[(51 - n):])
while 1:
    if method == methods:
        break
    else:
        for i in method_bin:
            if i == '1':
                position = goRight(blog, position)
            else:
                position = goRightOrLeft(blog, position)
        if position == [1, n - 1]:
            flag += 1
            # print(method_bin)
        position = [0, 0]
        method += 1
        method_bin = str('{:050b}'.format(method)[(51 - n):])

if flag == 0:
    flag = -1
print(flag)

# 输入:
# 20
# ....X..X..X.XX......
# ..XX.X..X......X....

# 输出:
# 256