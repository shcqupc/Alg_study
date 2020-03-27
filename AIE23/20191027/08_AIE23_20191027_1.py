print('\n----------------概率与统计--------------------')
import numpy as np
import scipy
import sympy as sym
import matplotlib.pyplot as plt
import sklearn.datasets as sd

iris = sd.load_iris()
x1 = np.random.random([10000])  # 均匀分布
x2 = np.random.normal(2, 1, [10000])  # 正态分布
x3 = np.random.normal(5, 1, [10000])  # 正态分布
# print(len(x1),len(x2))
# print(x1.shape,x2.shape)

# coin = np.random.randint(0, 3, [1000])
# print(coin)
# print(np.mean(coin))
# plt.hist(coin)
# plt.hist(x1, bins=20)

x1_mu = np.mean(x1)
x1_std = np.std(x1)
x2_mu = np.mean(x2)
x2_std = np.std(x2)
x3_mu = np.mean(x3)
x3_std = np.std(x3)

print('\n-----------------高斯分布-------------------')


def mode(x, mu, std):
    return 1 / np.sqrt(2 * np.pi) / std * np.exp(-(x - mu) ** 2 / 2 / std ** 2)


# xplot = np.linspace(-2, 8, 10000)
# print(x2_mu, x2_std, x3_mu, x3_std)
# x1_guass = mode(xplot, x1_mu, x1_std)
# x2_guass = mode(xplot, x2_mu, x2_std)
# x3_guass = mode(xplot, x3_mu, x3_std)
# plt.plot(xplot, x1_guass)
# plt.plot(xplot, x2_guass)
# plt.plot(xplot, x3_guass)
# plt.hist(x1, bins=30, alpha=0.5, density=True)
# plt.hist(x2, bins=30, alpha=0.5, density=True)
# plt.hist(x3, bins=30, alpha=0.5, density=True)
# plt.show()

print('\n---------------散点图---------------------')
x5 = np.random.normal(1, 1, [1000])
x6 = 2 * x5 + 1 + np.random.normal(0, 0.6, [1000])  # 噪声
# plt.scatter(x5, x6)
# plt.show()

print('\n-----------------协方差-------------------')
rand1 = np.random.normal(loc=1, scale=3, size=[1000]) * 10
rand2 = np.random.normal(1, 3, size=[1000]) * 10


# plt.hist(rand1, bins=30, alpha=0.5, density=True)
# plt.hist(rand2, bins=30, alpha=0.5, density=True)
# plt.show()

def conv(dt1, dt2):
    return np.mean((dt1 - np.mean(dt1)) * (dt2 - np.mean(dt2)))


print('conv', conv(x5, x6))

print('\n----------------线性相关系数--------------------')


def rho(p1, p2):
    return conv(p1, p2) / np.std(p1) / np.std(p2)


print('\n---------------坐标轴旋转---------------------')
print('rho', rho(x5, x6))
# plt.scatter(x5, x6)
# plt.axis("equal")
# plt.show()

print('\n---------------信息熵---------------------')
px = np.linspace(0.01, 0.99, 1000)
y = px * np.log((1 / px)) + (1 - px) * np.log(1 / (1 - px))
plt.plot(px, y)
plt.show()
