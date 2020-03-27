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

# (0) load training data
iris = load_iris()
X = iris.data
y = iris.target

# (1) test train split #
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# (2) Model training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# (3) Predict & Estimate the score
#y_pred = knn.predict(X_test)
print(knn.score(X_test, y_test))














