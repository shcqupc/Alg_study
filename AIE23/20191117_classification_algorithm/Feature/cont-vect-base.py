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
import sklearn.feature_extraction.text as t2v

text = ['纽约市 初步  初步 迹象 显示 初步',
        '初步 迹象 显示 这是 蓄意',
        '也 无 明确 证据 显示 迹象']
vectorizer = t2v.CountVectorizer()
vectors = vectorizer.fit_transform(text)
print("单词向量:\n", vectors.todense())
print("字典", vectorizer.vocabulary_)

