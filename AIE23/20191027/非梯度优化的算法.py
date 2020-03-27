import numpy as np 
def f(x1, x2):
    return x1**2+2*x1+2*x2**2 

x1 = 6 
x2 = 6 
for step in range(200):
    x1_t = x1 + np.random.normal(0, 0.1, [1]) 
    x2_t = x2 + np.random.normal(0, 0.1, [1])
    if f(x1, x2) > f(x1_t, x2_t):
        x1 = x1_t 
        x2 = x2_t 
        print(step, x1, x2, f(x1, x2))
