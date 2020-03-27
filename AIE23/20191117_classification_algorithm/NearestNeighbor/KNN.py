#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
==============================
KNN
==============================
KNN方法中没有训练过程，其分类方式就是寻找训练集附近的点。
所以带来的一个缺陷就是计算代价非常高
但是其思想实际上却是机器学习中普适的
"""
print(__doc__)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons, make_circles, make_classification
#引入训练数据
#X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
#X, y = make_moons(noise=0.1, random_state=1)
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, class_sep=0.2)
#定义随KNN分类器
knn = KNeighborsClassifier(n_neighbors=10)
#训练过程
knn.fit(X, y)
#绘图库引入
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
#调整图片风格
mpl.style.use('fivethirtyeight')
#定义xy网格，用于绘制等值线图
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
#预测可能性
Z = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("KNN")
plt.axis("equal")
plt.show()
