from numpy import linalg as LA
import numpy as np
import scipy
import sympy as sym
import matplotlib
import matplotlib.pyplot as plt

file = np.load("Resources/homework.npz")
print(file.keys())
X = file['X']
d = file['d']
print('X.shape:{}, d.shape:{}'.format(X.shape, d.shape))
# plt.scatter(X,d)
# plt.show()

"""
建模
假定数据形式为：y=ax+b
求loss函数 （y-d)**2 的最小值
"""


# 定义模型的函数
def mode(x, a, b):
    return a * x + b


# 定义梯度函数
def grad(x, d, a, b):
    y = mode(x, a, b)
    return 2 * (y - d) * x, 2 * (y - d)


# 梯度下降法
a = 5
b = 5
eta = 0.01
batch_size = 64
"""每次输入一个样本"""
for step in range(10000):
    sel_idx = np.random.randint(0, len(X))
    da, db = grad(X[sel_idx], d[sel_idx], a, b)
    a = a - eta * da
    b = b - eta * db
    #print("a:{}, b:{}, f(ax+b):{}, grad():{}".format(a, b, mode(X[sel_idx], a, b), grad(X[sel_idx], d[sel_idx], a, b)))

"""每次输入多个样本"""
# for step in range(200):
#     sel_idx = np.random.randint(0, len(X), batch_size)
#     inx = X[sel_idx]
#     ind = d[sel_idx]
#     da, db = grad(inx, ind, a, b)
#     a = a - eta * np.mean(da)
#     b = b - eta * np.mean(db)

xplt = np.linspace(-2, 5, 1000)
yplt = mode(xplt, a, b)
# 引入中文字库
matplotlib.rcParams['font.sans-serif'] = 'SimHei'
# 绘图
plt.scatter(X[:, 0], d[:, 0], s=20, alpha=0.4, label="数据散点")
plt.plot(xplt, yplt, lw=5, color="#990000", alpha=0.5, label="预测关系")
plt.legend()
plt.show()
