print('\n---------------one variable function---------------------')

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x ** 2 + 4 * x + 1


plt.figure(1)
x = np.linspace(-10, 10, 200)

plt.plot(x, x*0)
plt.plot(0, f(x))
plt.plot(x, f(x))
plt.show()


def df(x):
    return 2 * x + 4


x_old = 3.14
for itr in range(200):
    x_new = x_old - 0.3 * df(x_old)
    print("df={},  f({})={}".format(df(x_old), x_new, f(x_new)))
    x_old = x_new

print('\n---------------bivariate function---------------------')


def f_2d(x, y):
    return x ** 2 + 3 * x + y ** 2 + 8 * y + 1


def df_2d(x, y):
    return 2 * x + 2 * y + 11
