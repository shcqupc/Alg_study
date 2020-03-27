#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
MNIST-手写数字识别
=====================
对于简单的机器学习算法来说，数据预处理过程非常重要。
通过降维等操作可以使得运算过程极大的加快。
同时提高准确性
"""
print(__doc__)
import joblib
from sklearn.svm import SVC
import numpy as np
# from sklearn.externals import joblib
from sklearn.decomposition import PCA

train = np.load("../data/train.npz")
test = np.load("../data/test.npz")
pca = PCA(n_components=30)
X_r = pca.fit(train["images"]).transform(train["images"])
vect_t = np.array([[itr] for itr in range(10)])
X_train = X_r[:6000]
y_train = np.dot(train["labels"][:6000], vect_t).ravel().astype("int")
X_test = X_r[6000:12000]
y_test = np.dot(train["labels"][6000:12000], vect_t).ravel().astype("int")


#定义SVM分类器类
#lsvm = joblib.load("model/svm_model")
NBM = [SVC(kernel='linear', C=1.0, gamma = 'scale'),
       SVC(kernel='rbf', C=1.0, gamma = 'scale'),
       SVC(kernel='poly', C=1.0, gamma = 'scale')]
NAME= ["LINEAR","RBF","poly"]
for itr, itrname in zip(NBM, NAME):
    #训练过程
    print("Training...")
    itr.fit(X_train, y_train)
    print("Applying...")
    y_pdt = itr.predict(X_test)
    joblib.dump(itr, "../model/svm_model"+itrname)
    dts = len(np.where(y_pdt==y_test)[0])/len(y_test)

    print("{} 精度:{:.3f}".format(itrname, dts*100))