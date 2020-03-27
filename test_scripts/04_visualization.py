import matplotlib.pyplot as plt
import numpy as np
#
# x = np.linspace(-2, 2, 100)
# print(x)
# y = 3 * x + 1
# plt.plot(x, y)
# plt.show()

from mpl_toolkits.mplot3d import Axes3D
X = np.arange(-2,2,0.1)
Y = np.arange(-2,2,0.1)
X,Y= np.meshgrid(X, Y)
Z = -1 * (X + Y)

# 创建 3D 图形对象
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X,Y,Z,cmap=plt.cm.winter)
plt.show()