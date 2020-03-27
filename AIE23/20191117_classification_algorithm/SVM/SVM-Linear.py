#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
SVM算法
=====================
SVM算法时间复杂度取决于支持向量个数，虽然结果上与单层神经网络(逻辑回归)类似，但是实现原理并不相同
"""
print(__doc__)

from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
#引入训练数据
#X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, class_sep=1)
#定义SVM分类器类
lsvm = SVC(kernel='linear', C=10)
#训练过程
lsvm.fit(X, y)
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
Z = lsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
sv = lsvm.support_vectors_ 
plt.scatter(sv[:, 0], sv[:, 1], s=20, marker='^', color="#000099")
plt.title("SVM-linear")
plt.axis("equal")
plt.show()
