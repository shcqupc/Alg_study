# numpy官方教程： https://docs.scipy.org/doc/numpy-dev/user/quickstart.html 
# numpy官方教程中文翻译： NumPy的详细教程

#1. 创建数组和数组变形
import numpy as np
#
# 创建数组
a = np.array([1,2,3,4,5,6])
print(a)
# 直接给a.shape赋值是最简单的变形方式
a.shape = (2,3)
print('变形之后：')
print(a)

# [1 2 3 4 5 6]
# [[1 2 3]
#  [4 5 6]]

a.ravel() # 拉直数组

#array([1, 2, 3, 4, 5, 6])

#2.数组拼接
A = np.floor(np.random.randn(2,3) * 10)
print('A:\n', A)
B = np.floor(np.random.randn(2,3) * 10)
print('B:\n', B)

# A:
# [[ -2.   3. -10.]
#  [  5.   4.   7.]]
# B:
# [[-14.  -7.   3.]
#  [ 10.   6.  -8.]]

# 按第一个轴拼接
print('按行拼接：')
print(np.vstack([A,B]))
# 按第二个轴拼接
print('按列拼接：')
print(np.hstack([A,B]))

# 按行拼接：
# [[ -2.   3. -10.]
#  [  5.   4.   7.]
#  [-14.  -7.   3.]
#  [ 10.   6.  -8.]]
# 按列拼接：
# [[ -2.   3. -10. -14.  -7.   3.]
#  [  5.   4.   7.  10.   6.  -8.]]

#3. 基本操作和基本运算
np.exp(2)

#7.3890560989306504

np.exp2(2)

#4.0

np.sqrt(4)

#2.0

np.sin([2,3])

#array([ 0.90929743,  0.14112001])

np.log(2)

#0.69314718055994529

np.log10(2)

#0.3010299956639812

np.log2(2)

# 1.0

np.max([1,2,3,4])

#4

#4.二维数组完成矩阵操作
A = np.array([[1, 2], [-1, 4]])
B = np.array([[2, 0], [3, 4]])
print('对应元素乘：')
print(A * B)
print('矩阵乘法:')
print(np.dot(A, B)) # 或者 A.dot(B)
# 对应元素想乘：
# [[ 2  0]
#  [-3 16]]
# 矩阵乘法
# [[ 8  8]
#  [10 16]]

# 线性代数
from numpy import linalg

# 求A的转置
print('A的转置:')
print(A.transpose())

# 求A的逆矩阵
print('A的逆矩阵：')
print(linalg.inv(A))

# 特征值和特征向量
eigenvalues, eigenvectors = linalg.eig(A)
print('A 的特征值：')
print(eigenvalues) # 特征值
print('A 的特征向量：')
print(eigenvectors) # 特征向量

# A的转置:
# [[ 1 -1]
#  [ 2  4]]
# A的逆矩阵：
# [[ 0.66666667 -0.33333333]
#  [ 0.16666667  0.16666667]]
# A 的特征值：
# [ 2.  3.]
# A 的特征向量：
# [[-0.89442719 -0.70710678]
#  [-0.4472136  -0.70710678]]