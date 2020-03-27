#coding: UTF-8
# 不属于同一量纲：即特征的规格不一样，不能够放在一起比较。无量纲化可以解决这一问题。
# # 和MinMax相比适合整体都不太规整，方差较大的场景
# 公式为：(X-X_mean)/X_std 计算时对每个属性/每列分别进行.

# 将数据按其属性(按列进行)减去其均值，然后除以其方差。最后得到的结果是，对每个属性/每列来说所有数据都聚集在0附近，方差值为1。

# 首先说明下sklearn中preprocessing库里面的scale函数使用方法：

# sklearn.preprocessing.scale(X, axis=0, with_mean=True,with_std=True,copy=True)
# 1
# 根据参数的不同，可以沿任意轴标准化数据集。

# 参数解释：

# X：数组或者矩阵
# axis：int类型，初始值为0，axis用来计算均值 means 和标准方差 standard deviations. 如果是0，则单独的标准化每个特征（列），如果是1，则标准化每个观测样本（行）。
# with_mean: boolean类型，默认为True，表示将数据均值规范到0
# with_std: boolean类型，默认为True，表示将数据方差规范到1
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
X_train = np.arange(1,10,1).reshape(3,3)
X_test = X_train*10
print(X_train)
X_scaled1 = preprocessing.scale(X_train)
X_scaled2 = preprocessing.scale(X_train,axis=1)
print('\n---------------X_scaled1---------------------')
print(X_scaled1)
print('\n---------------X_scaled2---------------------')
print(X_scaled2)
print('\n---------------X_scaled2.mean---------------------')
print(X_scaled2.mean())
print('\n---------------X_scaled2---------------------')
print(X_scaled2.std())
print('\n-----------------StandardScaler-------------------')
st_scaler = StandardScaler(with_std=False).fit(X_train)
print("st_scaler.mean_",st_scaler.mean_)
print("st_scaler.scale_",st_scaler.scale_)
print('\n---------------transform(X_train)---------------------')
print(st_scaler.transform(X_train))
print("X_test",X_test)
print('\n----------------transform(X_test)--------------------')
print(st_scaler.transform(X_test))