#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
算法原理
=====================
对于方差不均等的高斯环境贝叶斯分类器而言，并不是一个单层的线性分类器。
但是我们依然可以用一个单层神经网络去完成分类任务。而且二者结果十分接近。
因此我们对神经网络进行了非线性处理，这一点上可以看到神经网络与高斯环境贝叶斯分类器的关联。
高斯环境下的贝叶斯分类器等同于一个单层神经网络，但是对于非线性可分问题，需要加入数据二次方项，以产生非线性结果。
所以一些文章中在简化情况(数据协方差矩阵相等)下将贝叶斯分类器总结成线性分类器。
"""
print(__doc__)

import tensorflow as tf
import tensorflow.contrib.slim as slim
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.datasets import make_moons, make_circles, make_classification
#获取数据
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
#X, y = make_moons(noise=0.3, random_state=0)
y_r = np.zeros([len(X), 2])
for idx, itr in enumerate(y):
    y_r[idx, itr] = 1
#搭建神经网络

x_tf = tf.placeholder(tf.float32, [None, 2])
label_tf = tf.placeholder(tf.float32, [None, 2])

x2x = tf.concat([x_tf, x_tf[:,:1]*x_tf[:,1:2], x_tf**2], axis=1)
y_tf = slim.fully_connected(x2x, 2, scope='full1', activation_fn=tf.nn.sigmoid, reuse=False)

ce=tf.nn.softmax_cross_entropy_with_logits(labels=label_tf, logits=y_tf)
loss = tf.reduce_mean(ce)
train_step = tf.train.AdamOptimizer(0.005).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for itr in range(6000):
    sess.run(train_step, feed_dict={x_tf: X, label_tf: y_r})


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
pdt = sess.run(y_tf, feed_dict={x_tf: np.c_[xx.ravel(), yy.ravel()]})
Z = pdt[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.6)

#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("NN")
plt.axis("equal")
plt.show()