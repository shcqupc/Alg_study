# 载入此项目所需要的库
import numpy as np
import pandas as pd
import visuals as vs # Supplementary code

# 载入波士顿房屋的数据集
data = pd.read_csv('2_boston/mlnd_boston_housing-master/housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# 完成
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

#TODO 1

#目标：计算价值的最小值
minimum_price = np.min(prices)

#目标：计算价值的最大值
maximum_price = np.max(prices)

#目标：计算价值的平均值
mean_price = np.mean(prices)

#目标：计算价值的中值
median_price = np.median(prices)

#目标：计算价值的标准差
std_price = np.std(prices)

#目标：输出计算的结果
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))

import matplotlib
rm = data['RM']
medv = data['MEDV']
matplotlib.pyplot.scatter(rm, medv, c='b')
matplotlib.pyplot.show()
lstat = data['LSTAT']
matplotlib.pyplot.scatter(lstat, medv, c='c')
matplotlib.pyplot.show()
ptratio = data['PTRATIO']
matplotlib.pyplot.scatter(ptratio, medv, c='g')
matplotlib.pyplot.show()
