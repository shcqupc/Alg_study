import numpy as np 
# 3*3随机矩阵 
A = np.random.random([3, 3]) 
B = np.random.random([3, 3])  
# 矩阵乘法
C = np.dot(A, B) 
# 哈达玛积 
C = A * B  
# 单位矩阵
C = np.eye(5)