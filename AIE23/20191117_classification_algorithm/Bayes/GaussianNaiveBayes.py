#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
高斯环境的贝叶斯分类器
=====================
朴素贝叶斯分类器是一种基于贝叶斯理论的简单的概率分类器，而朴素的含义是指输入变量的特征属性之间具有很强的独立性。
尽管这种朴素的设计和假设过于简单，但朴素贝叶斯分类器在许多复杂的实际情况下具有很好的表现，并且在综合性能上，该分类器要优于提升树（boosted trees）和随机森林（random forests）。
在许多实际应用中，对于朴素贝叶斯模型的参数估计往往使用的是极大似然法，因此我们可以这么认为，在不接受贝叶斯概率或不使用任何贝叶斯方法的前提下，我们仍然可以应用朴素贝叶斯模型对事物进行分类。
"""
print(__doc__)

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_moons, make_circles, make_classification
#引入训练数据
#X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
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
Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("GaussianNaiveBayes")
plt.axis("equal")
plt.show()
