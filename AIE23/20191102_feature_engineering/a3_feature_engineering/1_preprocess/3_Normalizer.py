#coding: UTF-8
# 不属于同一量纲：即特征的规格不一样，不能够放在一起比较。无量纲化可以解决这一问题。
# 服务于大量向量点乘运算的场景，防止因为向量值过大造成极端的影响。
# 归一化是依照特征矩阵的行处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准，也就是说都转化为“单位向量”。
# Normalization主要思想是对每个样本计算其p-范数，然后对该样本中每个元素除以该范数，这样处理的结果是使得每个处理后样本的p-范数（l1-norm,l2-norm）等于1。
#              p-范数的计算公式：||X||p=(|x1|^p+|x2|^p+...+|xn|^p)^1/p
from __future__ import print_function
import numpy as np
from sklearn.preprocessing import Normalizer
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = Normalizer()
print(scaler.fit(data))
print(scaler.transform(data))
print(scaler.transform([[2, 2]]))
