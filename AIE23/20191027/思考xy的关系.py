import numpy as np
import matplotlib.pyplot as plt

# 正态分布
x = np.random.normal(1, 1, [1000])
d = x ** 2 + 1 + np.random.normal(0, 0.6, [1000])


print('\n-----------------建立模型-------------------')
def model(x, a, b, c):
    return a * x ** 2 + b * x + c


def grad(x, d, a, b, c):
    y = model(x, a, b, c)
    return 2 * (y - d) * x ** 2, 2 * (y - d) * x, 2 * (y - d)


a = 0.1
b = 0.1
c = 0.0
eta = 0.02
batch_size = 32
for itr in range(1000):
    sel_idx = np.random.randint(
        0, len(x), batch_size)
    inx = x[sel_idx]
    ind = d[sel_idx]
    ga, gb, gc = grad(inx, ind, a, b, c)
    a = a - eta * np.mean(ga)
    b = b - eta * np.mean(gb)
    c = c - eta * np.mean(gc)
    print(a, b, c)

xplot = np.linspace(-2, 7, 1000)
yplot = model(xplot, a, b, c)
plt.scatter(x, d)
plt.plot(xplot, yplot, lw=2, c="r")
plt.show()
