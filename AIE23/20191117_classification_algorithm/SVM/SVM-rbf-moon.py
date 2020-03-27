#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
SVM算法
=====================
线性分类算法对于线性不可分问题解决较为复杂
"""
print(__doc__)

from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
#引入训练数据
#X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
X, y = make_moons(noise=0.3, random_state=0)
#定义SVM分类器类
#随着gamma增大，细节越丰富，这样注意过拟合问题
lsvm = SVC(kernel='rbf', gamma=1.0, C=1.0)
#训练过程rbf
lsvm.fit(X, y)
#绘图库引入rbf
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
#Z = lsvm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
sv = lsvm.support_vectors_ 
plt.scatter(sv[:, 0], sv[:, 1], s=60, marker='^', color="#990099")
plt.title("SVM-rbf-cross")
plt.axis("equal")
plt.show()
