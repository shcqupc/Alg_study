#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
==============================
文本向量化方法1
==============================
统计词频
"""
print(__doc__)
from sklearn.datasets import fetch_20newsgroups
import sklearn.feature_extraction.text as t2v
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import numpy as np
#获取数据
newsgroups_train = fetch_20newsgroups(data_home="data", subset='train')
newsgroups_test = fetch_20newsgroups(data_home="data", subset = 'test')
#单词向量化
vectorizer = t2v.CountVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)

print("class:", newsgroups_train.target_names[newsgroups_train.target[1]])
print("data:", newsgroups_train.data[1])
print(vectors[1])
