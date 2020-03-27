import numpy as np
import scipy
import sympy as sym
import matplotlib.pyplot as plt

sym.init_printing()

print('\n-------------------求一维函数最小值-----------------')


def f(x):
    """原函数"""
    # return x ** 2 + 1
    return np.sin(x)


def df(x):
    """导数"""
    # return 2 * x
    return np.cos(x)


x = 0
eta = 0.02
for s in range(200):
    dx = eta * df(x)
    x = x - dx
    # print("s:{}, x:{:.2f}, f(x):{:.2f}, df(x):{:.2f}".format(s, x, f(x), df(x)))

print('\n-------------------求二维函数最小值-----------------')


def f(x1, x2):
    return x1 ** 2 + 2 * x1 + 2 * x2 ** 2


def df(x1, x2):
    return 2 * x1 + 2, 4 * x2


x1 = 6
x2 = 6
eta = 0.15
for s in range(20):
    g1, g2 = df(x1, x2)
    x1 = x1 - eta * g1
    x2 = x2 - eta * g2
    # print("s:{}, x1:{:.2f}, x2:{:.2e}, f():{:.2e}, df():{}".format(s, x1, x2, f(x1, x2), list(df(x1, x2))))

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


print('\n---------------3*3 matrix---------------------')
A = np.random.random([3, 3])
print(A.shape)
print('\n--------------matrix by matrix----------------------')
B = np.random.random([3, 3])
C = np.dot(A, B)
D = A * B
E = np.eye(3)
print(C.shape)
print(D.shape)
print('E:{}'.format(E))
print(D * E)

print('\n---------------仿射变换---------------------')
x = np.random.random([1000, 2])
print(x)
A = np.array([[5, 1], [-0.5, 1]])
# print(A.shape)
y = np.dot(x, A)
# print(y)
plt.scatter(x[:, 0], x[:, 1], c='r')
plt.scatter(y[:, 0], y[:, 1], c='b')
plt.axis("equal")
plt.show()
