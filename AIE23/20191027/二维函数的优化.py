import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 定义函数
def f(x1, x2):
    return x1 ** 2 + 2 * x1 + 2 * x2 ** 2


# 定义函数的偏导数
def df(x1, x2):
    return 2 * x1 + 2, 4 * x2

grad = (6,6)
x1 = 6
x2 = 6
eta = 0.05  # 步长
# 迭代使函数值不断变小
for step in range(200):
    g1, g2 = df(x1, x2)
    x1 = x1 - eta * g1
    x2 = x2 - eta * g2
    print(step, g1, g2, x1, x2, f(x1, x2))

# 绘制函数的3D曲线
fig = plt.figure(figsize=(4.5,3))
ax = Axes3D(fig)
x1 = np.linspace(-10, 10, 1000)
x2 = np.linspace(-10, 10, 1000)
ax.plot3D(x1, x2, f(x1, x2), 'gray')
plt.show()
