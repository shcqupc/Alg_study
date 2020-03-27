import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print('\n----------------另一种建模方式--------------------')

# 读取数据
file = np.load("Resources/homework.npz")
print(file.keys())
X = file['X']
d = file['d']
print('X.shape:{}, d.shape:{}'.format(X.shape, d.shape))


# 观察散点图
# plt.scatter(X, d)
# plt.show()


# 定义非线性函数
def func(x):
    ret = np.array(x)
    ret[x < 0] = 0
    return ret


def dfunc(x):
    ret = np.zeros_like(x)
    ret[x > 0] = 1
    return ret


def f(x, w):
    a, b = w
    return func(a * x + b)


# 定义函数关于可训练参数的偏导数
def grad_f(x, d, w):
    a, b = w
    y = f(x, w)
    dy = dfunc(a * x + b)
    grad_a = 2 * (y - d) * dy * x
    grad_b = 2 * (y - d) * dy
    return grad_a, grad_b


# 梯度下降法
w = [0.1, 0.1]
eta = 0.01
batch_size = 64
"""每次输入多个样本"""
for step in range(200):
    sel_idx = np.random.randint(0, len(X), batch_size)
    inx = X[sel_idx]
    ind = d[sel_idx]
    da, db = grad_f(inx, ind, w)
    w[0] -= eta * np.mean(da)
    w[1] -= eta * np.mean(db)

xplt = np.linspace(-2, 5, 1000)
yplt = f(xplt, w)
# 引入中文字库
matplotlib.rcParams['font.sans-serif'] = 'SimHei'
# 绘图
plt.scatter(X[:, 0], d[:, 0], s=10, alpha=0.4, label="数据散点")
plt.plot(xplt, yplt, lw=5, color="#990000", alpha=0.5, label="预测关系")
plt.legend()
plt.show()
