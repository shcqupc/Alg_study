#coding: UTF-8
# 信息利用率低：不同的机器学习算法和模型对数据中信息的利用是不同的，之前提到在线性模型中，使用对定性特征哑编码可以达到非线性的效果。类似地，对定量变量多项式化，或者进行其他的转换，都能达到非线性的效果。
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
# a,b -> a, b, ab, a^2, b^2, 1
X = np.arange(9).reshape(3, 3)
print(X[0:5])

poly = PolynomialFeatures(2)
print(poly.fit_transform(X[0:5]))

# poly = PolynomialFeatures(interaction_only=True)
# print(poly.fit_transform(X))
