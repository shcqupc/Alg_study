import numpy as np 
import matplotlib.pyplot as plt 
# 正态分布
x1 = np.random.random([1000]) 
x2 = np.random.normal(3, 1, [1000]) 
# 随机生成01数字（概率相同）
conin = np.random.randint(0, 2, [1000000]) 
mean_coin = np.mean(conin) 
print(conin, mean_coin)
# 均匀分布
## 可以有LCG 
rand = np.random.random([100000]) 

# 高斯分布、正态分布 
rand = np.random.normal(0, 1, [10000])
rand2 = np.random.normal(0, 2, [10000])

#plt.hist(x1) 
#plt.hist(x2) 

def model(x, mu, std):
    """
    高斯分布
    """
    return 1/np.sqrt(2*np.pi)/std*np.exp(
        -(x-mu)**2/2/std**2
    )
xplot = np.linspace(-2, 7, 1000) 
x1_mu = np.mean(x1) 
x1_std = np.std(x1) 
x2_mu = np.mean(x2) 
x2_std = np.std(x2) 

x1_gauss = model(xplot, x1_mu, x1_std) 
x2_gauss = model(xplot, x2_mu, x2_std)

plt.plot(xplot, x1_gauss) 
plt.plot(xplot, x2_gauss)
plt.hist(x1, bins=30, alpha=0.5, normed=True)
plt.hist(x2, bins=30, alpha=0.5, normed=True)
plt.show()