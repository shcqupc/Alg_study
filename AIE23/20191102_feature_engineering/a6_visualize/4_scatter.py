"""
Please note, this script is for python3+.
If you are using python2+, please modify it accordingly.
Tutorial reference:
http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
"""

import matplotlib.pyplot as plt
import numpy as np

# X = np.linspace(-3, 3, 50)
# print(X)
# Y = 2 * X + 1
# # print(Y)
# T = Y    # for color later on
# print(T)
#plt.scatter(X, Y, c=T)
x = [0, 1, 2]
y = [11, 22, 33]
plt.bar(x, y)
# #plt.xlim(-1.5, 1.5)
# #plt.xticks(())  # ignore xticks
# #plt.ylim(-1.5, 1.5)
# #plt.yticks(())  # ignore yticks
plt.show()

