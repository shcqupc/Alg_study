from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

d = load_iris().data
t = load_iris().target

print('\n-----------------np.mean--np.std-----------------')
print(np.mean(d, axis=0))
print(np.std(d, axis=0))
print(d[:5])
print('\n----------------Scaler--------------------')
from sklearn import preprocessing as pp

d1 = pp.scale(d)
print(np.mean(d1, axis=0))
print(np.std(d1, axis=0))
print(d1[:5])
plt.figure(figsize=[5, 3])
plt.hist(d[:, 1], bins=15)
plt.hist(d1[:, 1], bins=15)
plt.show()

print('\n----------------StandardScaler--------------------')
# from sklearn import preprocessing as pp
# d1 = pp.StandardScaler().fit_transform(d,t)
# print(np.mean(d1, axis=0))
# print(np.std(d1, axis=0))
# print(d1[:5])
# plt.figure(figsize=[5,3])
# plt.hist(d[:,1],bins=10)
# plt.hist(d1[:,1],bins=10)
# plt.show()

print('\n----------------MinMaxScaler--------------------')
# from sklearn import preprocessing as pp
# d1 = pp.MinMaxScaler().fit_transform(d,t)
# print('mean',np.mean(d1, axis=0))
# print('std',np.std(d1, axis=0))
# print(d1[:5])
# plt.figure(figsize=[5,3])
# plt.hist(d[:,1],bins=10)
# plt.hist(d1[:,1],bins=10)
# plt.show()


print('\n----------------Normalizer--------------------')
# from sklearn.preprocessing import Normalizer
# d1 = Normalizer().fit_transform(d,t)
# print('mean',np.mean(d1, axis=0))
# print('std',np.std(d1, axis=0))
# print(d1[:5])
# plt.figure(figsize=[5,3])
# plt.hist(d[:,1],bins=10)
# plt.hist(d1[:,1],bins=10)
# plt.show()

print('\n----------------Binarizer--------------------')
# from sklearn.preprocessing import Binarizer
# d1 = Binarizer(threshold=3).fit_transform(d)
# print(d1[:5])
# plt.figure(figsize=[5,3])
# plt.hist(d[:,1],bins=15)
# plt.hist(d1[:,1],bins=25)
# plt.show()

print('\n--------------PolynomialFeatures----------------------')
# from sklearn.preprocessing import PolynomialFeatures
# d1 = PolynomialFeatures(2).fit_transform(d)
# print(d1[:5])
# plt.figure(figsize=[5,3])
# plt.hist(d[:,1],bins=15)
# plt.hist(d1[:,1],bins=25)
# plt.show()

print('\n---------------Feature Selection---------------------')
print('\n--------------Removing features with low variance----------------------')
# print(np.var(d, axis=0))
# from sklearn.feature_selection import VarianceThreshold
# d1 = VarianceThreshold(0.6).fit_transform(d)
# print(d1[:5])

print('\n--------------SelectKBest(chi2)----------------------')
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# d1 = SelectKBest(chi2,k=2).fit_transform(d,t)
# print(d1[:5])

print('\n--------------RFE---------------------')
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# d1 = RFE(estimator=LogisticRegression(),n_features_to_select=2).fit_transform(d,t)
# print(d1[:5])

print('\n-----------------SelectFromModel------------------')
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression
# d1 = SelectFromModel(estimator=LogisticRegression(penalty="l1", C=0.1)).fit_transform(d,t)
# print(d1[:5])

print('\n-----------------GBDT------------------')
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import GradientBoostingClassifier
# d1 = SelectFromModel(estimator=GradientBoostingClassifier()).fit_transform(d,t)
# print(d1[:5])


print('\n---------------training---------------------')
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.model_selection import train_test_split as tts
# from sklearn.neighbors import KNeighborsClassifier
#
# from sklearn.model_selection import cross_val_score
#
# d1 = PolynomialFeatures(2).fit_transform(d)
# d_train, d_tst, t_train, t_tst = tts(d1, t, random_state=4)
# print(d_train.shape, d_tst.shape)
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(d_train, t_train)

# Predict & Estimate the score
# p = knn.predict(d_tst)
# print(p)
# print(t_tst)
# score = cross_val_score(estimator=knn, X=d1, y=t, cv=3, scoring="accuracy")
# print(score)
# from sklearn.externals import joblib
# # joblib.dump(knn, '../../a4_model_load/model/save/knn1.pkl')
# knn1 = joblib.load('../../a4_model_load/model/save/knn1.pkl')
# score = cross_val_score(estimator=knn1, X=d1, y=t, cv=5, scoring="accuracy")
# print(score)
# k_range = range(1, 31)
# k_score = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, d1, t, cv=10, scoring='accuracy')
#     k_score.append(scores.mean())
#
# plt.plot(k_range, k_score)
# plt.xlabel('Value of k for knn')
# plt.ylabel('Cross-validated Accuracy')
# plt.show()
