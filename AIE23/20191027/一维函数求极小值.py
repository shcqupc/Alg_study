import numpy as np 
def f(x):
    """定义函数"""
    return np.sin(x)
def df(x):
    """函数的导数"""
    return np.cos(x)

x = 0 
eta = 0.1 
for step in range(20):
    dx = eta * df(x)
    x = x - dx 
    print("Step:{}, x:{:.2f},f(x):{:.2f}".format(step, x, f(x)))
