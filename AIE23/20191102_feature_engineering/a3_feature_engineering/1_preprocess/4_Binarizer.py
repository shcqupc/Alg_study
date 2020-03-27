#coding: UTF-8
# 信息冗余：对于某些定量特征，其包含的有效信息为区间划分，例如学习成绩，假若只关心“及格”或不“及格”，那么需要将定量的考分，转换成“1”和“0”表示及格和未及格。二值化可以解决这一问题。
from sklearn.preprocessing import Binarizer
import numpy as np

Y = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
X = np.array([[-3., 5., 15],
              [-1., 6., 14],
              [6., 3., -11]])
binarizer = Binarizer().fit(X)  # fit does nothing
print(binarizer.transform(X))
print(binarizer.transform(Y))

