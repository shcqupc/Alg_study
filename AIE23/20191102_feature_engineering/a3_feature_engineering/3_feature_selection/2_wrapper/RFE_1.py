from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np

d = load_iris().data
t = load_iris().target
# selector = RFE(estimator=LogisticRegression(),n_features_to_select=2).fit(d,t)
# print(selector.transform(d))