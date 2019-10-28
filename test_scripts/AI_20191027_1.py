import numpy as np
import matplotlib as plt

print('\n---------------3*3 matrix---------------------')
A = np.random.random([3, 3])
print(A)
print('\n--------------matrix by matrix----------------------')
B = np.random.random([3, 3])
C = np.dot(A, B)
D = A * B
print(C)
print(D)

print('\n---------------仿射变换---------------------')
x = np.round(np.random.random([10, 2]))
print(x)
E = np.array([[5, 0], [0, 1]])
y = np.dot(x, E)
print(E)
print(y)
