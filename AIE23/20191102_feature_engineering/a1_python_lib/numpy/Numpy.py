import numpy as np
from matplotlib import pyplot as plt

# 一维
a = np.array([[1,2], [2,4], [3,3]]); 
print(a) # [1 2 3]
# 等间隔数字的数组
b = np.arange(10); 
# b = np.arange(16); [0, 1, 2  ... ]
# reshape -> 4*4

print(b) # [0 1 2 3 4 5 6 7 8 9]
b = np.arange(3,7,2);
print(b) # array([3, 5])
# 二维
c = np.array([[1, 2], [3, 4]]); 
print(c) # [[1 2]
								#  [3 4]]

c = np.array([[1, 2, 3], [3, 4, 2], [3, 4, 2]]); 
d = np.array([1,2,3])
							
print(c)
print(c.shape)
print(c[0:10])

# 调整数组shape
a = np.array([[1, 2, 3], [4, 5, 6]]); print(a); # [[1 2 3]
												#  [4 5 6]]
a.shape = (3, 2); print(a)                      # [[1 2]
												#  [3 4]
												#  [5 6]]

# ndim:返回数组的维数
a = np.arange(24);
print(a)
print(a.ndim) # 1

# numpy.reshape: 在不改变数据的条件下修改形状
b = a.reshape(2, 4, 3); 
print(b.ndim) # 3

# 空数组
x = np.empty([3, 2], dtype='i1'); print(x) # 数组x的元素为随机值，因为它们未初始化

# 含有5个0的数组，若不指定类型，则默认为float
x = np.zeros(5, dtype=np.int); print(x) # [0 0 0 0 0]
# 含有6个1的二维数组，若不指定类型，则默认为float
x = np.ones([2, 3], dtype=int); print(x) # [[1 1 1]
										 #  [1 1 1]]

# 使用内置的range()函数创建列表对象
x = range(5); print(x) # range(0, 5)

# numpy.linspace，此函数类似于arange，在此函数中，指定了范围之间的均匀间隔数量，而不是步长


# 算数运算：add, subtract, multiply, divide, reciprocal, power, mod 输入数组必须具有相同的形状或符合数组广播规则
a, b = [5, 6], [7, 10]
c = np.subtract(a, b); print(c) # [-2 -4]

# 统计函数：用于从数组中给定的元素中查找最小，最大，百分标准差和方差等, amin, amax, ptp, percentile, median, mean, average, std
a = np.array([1, 2, 3, 4, 5])
print(np.amin(a)) # 1
print(np.mean(a)) # 3.0
