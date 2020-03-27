#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
贝叶斯分类器用于鸢尾花数据
=====================
由Fisher在1936年整理，包含4个特征（Sepal.Length（花萼长度）、Sepal.Width（花萼宽度）、Petal.Length（花瓣长度）、Petal.Width（花瓣宽度））
特征值都为正浮点数，单位为厘米。目标值为鸢尾花的分类（Iris Setosa（山鸢尾）、Iris Versicolour（杂色鸢尾），Iris Virginica（维吉尼亚鸢尾））。
"""
print(__doc__)


from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA


iris = datasets.load_iris()
#选择两个属性，便于绘图
X = iris.data
y = iris.target

pca = PCA(n_components=2)
X = pca.fit(X).transform(X)


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
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
#预测可能性
pdt = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])

Z = pdt[:, 0]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.6)
Z = pdt[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.6)
#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("GaussianNaiveBayes")
plt.axis("equal")
plt.show()