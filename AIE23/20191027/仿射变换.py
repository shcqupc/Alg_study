import numpy as np 
import matplotlib.pyplot as plt  

x = np.random.random([1000, 2]) 
A = np.array([[1, 0.5], [-0.5, 2]]) 
y = np.dot(x, A) 
plt.scatter(x[:, 0], x[:, 1], c="r")
plt.scatter(y[:, 0], y[:, 1], c="b")
plt.axis("equal")
plt.show()