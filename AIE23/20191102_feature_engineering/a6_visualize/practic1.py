import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=[5,3])
X = np.linspace(1,10,1000)
Y = X**3 +10

plt.plot(X,Y, c="red", lw=2)
plt.show()

X = np.linspace(-2, 2, 20)
Y = 2 * X + 1
plt.scatter(X,Y, c='r')
plt.show()