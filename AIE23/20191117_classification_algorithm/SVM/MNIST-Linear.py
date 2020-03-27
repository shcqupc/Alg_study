#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
MNIST-手写数字识别
=====================
单层神经网络可以很好的完成手写数字的分类，同样SVM也可以
"""
print(__doc__)

from sklearn.svm import SVC
import numpy as np

train = np.load("../data/train.npz")
test = np.load("../data/test.npz")

vect_t = np.array([[itr] for itr in range(10)])
X_train = train["images"][:6000]
y_train = np.dot(train["labels"][:6000], vect_t).ravel().astype("int")
X_test = test["images"][:6000]
y_test = np.dot(test["labels"][:6000], vect_t).ravel().astype("int")


#定义SVM分类器类
lsvm = SVC(kernel='linear', C=1.0)
#训练过程
lsvm.fit(X_train, y_train)
print(len(lsvm.support_vectors_)) #支持向量的个数
y_pdt = lsvm.predict(X_test)

dts = len(np.where(y_pdt==y_test)[0])/len(y_test)

print("{} 精度:{:.3f}".format("Line", dts*100))