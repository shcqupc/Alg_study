import numpy as np
import matplotlib.pyplot as plt

print('\n---------------3*3 matrix---------------------')
A = np.random.random([3, 3])
print(A.shape)
print('\n--------------matrix by matrix----------------------')
B = np.random.random([3, 3])
# 矩阵相乘
C = np.dot(A, B)
# 哈达玛积
D = A * B
# 单位矩阵
E = np.eye(3)
print(C.shape)
print(D.shape)
print('E:{}'.format(E))
print(D * E)

print('\n---------------仿射变换---------------------')
import numpy as np
import matplotlib.pyplot as plt
x = np.random.random([1000, 2])
print(x)
A = np.array([[5, 0.5], [-0.5, 1]])
# print(A.shape)
y = np.dot(x, A)
# print(y)
plt.scatter(x[:, 0], x[:, 1], c='r')
plt.scatter(y[:, 0], y[:, 1], c='b')
plt.axis("equal")
plt.show()
