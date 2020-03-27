#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
==============================
Bootstrap aggregating(bagging)
==============================
随机森林算法属于ensemble算法的bagging分支
其核心思想就是将一些方差比较大(比较容易过拟合)的分类器进行组合
用随机的方式消除方差。
"""
print(__doc__)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons, make_circles, make_classification
#引入训练数据
#X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
X, y = make_moons(noise=0.1, random_state=1)
#定义随机森林分类器
rdf = RandomForestClassifier()
#训练过程
rdf.fit(X, y)
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
Z = rdf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("RandomForest")
plt.axis("equal")
plt.show()
