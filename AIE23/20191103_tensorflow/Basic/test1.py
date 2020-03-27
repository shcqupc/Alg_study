# import numpy as np
#
# A = np.reshape(np.linspace(1, 4, 4), [2, 2])
# print(A)
# print(A * A)
# print(np.dot(A, A))
#
# B = np.reshape(np.linspace(1, 4, 4), [4, 1])
# C = np.reshape(np.linspace(1, 4, 4), [1, 4])
# print(B)
# print(C)
# print(np.dot(B, C))
# print(B*C)
#
# x = np.linspace(-5, 5, 1000)
# y = np.sin(x)
# import matplotlib.pyplot as plt
# plt.plot(x,y, lw=5, c="y")
# plt.plot(x*0, y)
# plt.plot(x, y*0)
# plt.show()

import datetime
import numpy as np
print(datetime.datetime.now())
print(np.linspace(1,100,10))
print(np.random.randint(1,100,8))

tx = np.random.random([100, 1])
ty = tx + 1
print([[itr,(x_in, y_in)] for itr, (x_in, y_in) in enumerate(zip(tx, ty))])
print([[itr,x_in, y_in] for itr, x_in, y_in in enumerate(tx, ty)])

sigma = np.diag([18.54,1.83,5.01])
print(sigma)
print(np.dot(sigma,sigma))