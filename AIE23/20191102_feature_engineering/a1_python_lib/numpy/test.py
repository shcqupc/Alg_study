import numpy as np
# a = np.array([1, 2, 3]); 
# print(a)
# b = a * 2
# print(b)
# b = a / 255
# print(b)
# c = np.max(a)
# print(c)
# b = np.arange(10); 
# print(b)

# ndim:返回数组的维数
a = np.arange(24);
print(a)
print(a.ndim) # 1
# numpy.reshape: 在不改变数据的条件下修改形状
b = a.reshape(2, 4, 3); 
print(b)
print(b.ndim) # 3



