import numpy as np 
import matplotlib.pyplot as plt 
# 正态分布
x1 = np.random.normal(0, 1, [1000]) 
#x2 = 2 * x1 + 1 + np.random.normal(0, 0.6, [1000])
x2 = np.random.normal(0, 4, [1000])
# 协方差
def conv(x1, x2):
    return np.mean((x1-np.mean(x1))*(x2-np.mean(x2))) 
# 线性相关系数
def rho(x1, x2):
    return conv(x1, x2)/np.std(x1)/np.std(x2)
print(rho(x1, x2))

plt.scatter(x1, x2) 
plt.axis("equal")
plt.show() 