#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
高斯环境的贝叶斯分类器
=====================
在训练数据协方差相同的情况下，高斯分类器等效于线性分类器
文章：https://zhuanlan.zhihu.com/p/30824582
"""
print(__doc__)

from sklearn.naive_bayes import GaussianNB
import numpy as np
#引入训练数据
X1 = np.random.normal(size = [600, 2])
X2 = np.random.random([600, 2])
#方差均衡，使得两个类方差均为1
X1 = X1/np.std(X1)
X2 = X2/np.std(X2)
y = np.concatenate([np.zeros_like(X1[:,0]), np.ones_like(X2[:,0])], axis=0)
X = np.concatenate([X1, X2],axis=0)

#定义高斯分类器类
gnb = GaussianNB()
#训练过程
gnb.fit(X, y)
#绘图库引入
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
#调整图片风格
mpl.style.use('fivethirtyeight')
#定义xy网格，用于绘制等值线图
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001),
                     np.arange(y_min, y_max, 0.001))
#预测可能性
Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("GaussianNaiveBayes")
plt.axis("equal")
plt.show()
