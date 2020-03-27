print('\n---------------bivariate function---------------------')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f_2d(x, y):
    return x ** 2 + 3 * x + y ** 2 + 8 * y + 1


def df_2d(x, y):
    return 2 * x + 3, 2 * y + 8


x, y = 10, 10
for itr in range(50):
    dx, dy = df_2d(x, y)
    x, y = x - 0.3 * dx, y - 0.3 * dy
    print("dx={},dy={},  f({},{})={}".format(dx, dy, x, y, f_2d(x, y)))

x = np.linspace(-10, 10, 50)
y = np.linspace(-15, 10, 50)
X, Y = np.meshgrid(x, y)
ax = Axes3D(plt.figure())
surf = ax.plot_surface(X, Y, f_2d(X, Y), rstride=1, cstride=1, cmap='rainbow')
plt.show()




