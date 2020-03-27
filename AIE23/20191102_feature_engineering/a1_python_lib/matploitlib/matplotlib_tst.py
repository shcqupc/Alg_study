import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
# x = np.linspace(0,10,1000)
# y= 2*x**2+5
# plt.figure(figsize=(4,3))
# plt.plot(x,y,label="absadf",color = "red")
# plt.xlabel("x")
# plt.ylabel("sin(x)")
# plt.show()

x = np.random.random([1000])
plt.hist(x,bins=30)
plt.show()
