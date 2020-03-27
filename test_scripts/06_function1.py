print('\n---------------one variable function---------------------')

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x ** 2 + 4 * x + 1


def df(x):
    return 2 * x + 4


x_old = 3.14
for itr in range(50):
    x_new = x_old - 0.3 * df(x_old)
    print("df={},  f({})={}".format(df(x_old), x_new, f(x_new)))
    x_old = x_new

plt.figure(1)
x = np.linspace(-10, 10, 200)
plt.plot(x, x*0)
plt.plot(0*f(x), f(x))
plt.plot(x, f(x))
plt.show()



