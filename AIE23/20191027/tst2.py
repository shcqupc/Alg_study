import numpy as np
import matplotlib.pyplot as plt


print('\n---------------已知数据{(x,d)}，求数据间的关系---------------------')
x = np.random.normal(1, 1, [1000])
d = x ** 2 + 1 + np.random.normal(0, 0.6, [1000])

# plt.scatter(x, d)
# plt.show()


print('\n-----------------建立模型-------------------')


def mode(x, a, b, c):
    return a * x ** 2 + b * x + c


"""梯度函数"""


def grad(x, d, a, b, c):
    y = mode(x, a, b, c)
    return 2 * (y - d) * x ** 2, 2 * (y - d) * x, 2 * (y - d)


a = 0.1
b = 0.2
c = 0.3
eta = 0.02
batch_size = 100
for s in range(20):
    sel_idx = np.random.randint(0, len(x), batch_size)
    inx = x[sel_idx]
    ind = d[sel_idx]
    ga, gb, gc = grad(inx, ind, a, b, c)
    a = a - eta * np.mean(ga)
    b = b - eta * np.mean(gb)
    c = c - eta * np.mean(gc)

xplt = np.linspace(-2, 7, 1000)
yplt = mode(xplt, a, b, c)
plt.scatter(x, d)
plt.plot(xplt, yplt, lw=2, c='purple')
plt.show()