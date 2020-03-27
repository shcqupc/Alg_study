import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[4, 5], [6, 7], [8, 9]])
#print(A * B)
print(np.dot(A, B))

"""
x1 = np.random.normal(0, 1, [3])
x2 = np.random.normal(2, 1, [3])
plt.figure(figsize=(4, 3))
# bins: 柱形数量， alpha：透明度， density：是否归一化
# plt.hist(x1, bins=20,alpha=0.5, density=False)
# plt.hist(x2, bins=20,alpha=0.5, density=False)
print(np.log(8), np.log2(8), np.log10(1000), np.exp(2), np.e)
# y1 = x1**2
# y2 = np.log(x2)
# plt.plot(x1,y1)
# plt.plot(x2,y2)
# plt.show()


import sympy as sym

x = sym.symbols('x')
y = sym.symbols('y')
fx = x*3+9
print(sym.solve(fx, x))

"""
