"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
#coding: UTF-8
#Definiation of COLs:
#1. sepal length in cm
#2. sepal width in cm
#3. petal length in cm
#4. petal width in cm
#5. class:
#      -- Iris Setosa
#      -- Iris Versicolour
#      -- Iris Virginica
#Missing Attribute Values: None

from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures

iris = load_iris()
X = iris.data
y = iris.target
print(X, y)
# (0) feature engineering
# the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
poly = PolynomialFeatures(2)
X_Poly = poly.fit_transform(X)
#X = X_Poly
print(X_Poly)

# (1) test train split #
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)

# (2) Model training
knn.fit(X_train, y_train)

# (3) Predict & Estimate the score
y_pred = knn.predict(X_test)
print(knn.score(X_test, y_test))

# (4) Cross Validation
#  this is cross_val_score #
from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
print(scores)

# (5) Parameter tuining:
# this is how to use cross_val_score to choose model and configs #
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
##    loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error') # for regression
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy') # for classification
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()













