#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
文本分类实例
=====================
关键步骤是单词向量化
1.多项式模型
如果我们考虑重复词语的情况，也就是说，重复的词语我们视为其出现多次，
直接按条件独立假设的方式推导，这样的模型叫作多项式模型。
2.伯努利模型
另一种更加简化的方法是将重复的词语都视为其只出现1次，
这样的模型叫作伯努利模型（又称为二项独立模型）。这种方式更加简化与方便。
当然它丢失了词频的信息，因此效果可能会差一些。
"""
print(__doc__)


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import numpy as np
#获取数据
newsgroups_train = fetch_20newsgroups(data_home="data", subset='train')
newsgroups_test = fetch_20newsgroups(data_home="data", subset = 'test')
#单词向量化
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)

NBM = [MultinomialNB(alpha=0.01), BernoulliNB(alpha=0.01)]
NAME= ["多项式","伯努利"]
for itr, itrname in zip(NBM, NAME):
    itr.fit(vectors, newsgroups_train.target)
    pred = itr.predict(vectors_test)
    dts = len(np.where(pred==newsgroups_test.target)[0])/len(newsgroups_test.target)
    print("{} 精度:{:.3f}".format(itrname,dts*100))