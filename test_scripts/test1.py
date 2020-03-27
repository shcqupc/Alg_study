def team(a, *b):
    print(a + ':' + str(b))


team('A', 'B', 'C', 'D')



import numpy as np
import matplotlib.pyplot as plt
import math

# x = np.linspace(-1, 1, 1000)
# y = np.linspace(-1, 1, 1000)
# x, y = np.meshgrid(x, y)
# plt.contour(x, y, x ** 2 + y ** 2, [1])
print(np.pi)
# plt.figure(1)
# plt.subplot(211)
print(np.e)
print(np.log10(1000))
print(math.log(1000, 10))

# x = np.linspace(-1, 1, 1000)
# y = np.sinh(x)
# z = np.cosh(x)

# y = (np.e ** x - np.e ** (-x)) / 2
# z = (np.e ** x + np.e ** (-x)) / 2
# j = y / z

# y = np.sin(x)
# z = np.sin(y)
# j = x

# y = 0.5**x
# z = 2**x

# x = np.linspace(0, 1, 1000)
# y = 1/np.sqrt(x)
# x = np.linspace(0, 1, 1000)
# y = np.log2(x)
# z = math.log(x, 2)

# x = np.linspace(1, 100, 100,axis=0)
# y = ((-1)**x)*1/x

x = np.linspace(-10, 10, 2000)
y = np.sin(x)
plt.plot(x, y)
hx = x
hy = hx * 0
vy = np.linspace(-2, 2, 100)
vx = vy * 0
plt.plot(hx, hy)
plt.plot(vx, vy)


# plt.plot(x, z)
# plt.plot(x, j)
plt.show()
print(np.sin(np.pi/2))