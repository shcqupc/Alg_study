from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split  # former cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
import pyperclip as cp

# Feature Engineering
iris = load_iris()
x = iris['data']
y = iris.target
# print(x, y)
poly = PolynomialFeatures(2)
x_poly = poly.fit_transform(x)
print('x_poly', x_poly)
str = str(x_poly)
cp.copy(str)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)
print(x_train, y_train)

# Model Training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# Predict & Estimate the score
y_pred = knn.predict(x_test)
print('\n-----------y_pred--------------')
print(y_pred)
print('\n-----------y_test--------------')
print(y_test)

print('\n-----------cross_val_score--------------')
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

knn = KNeighborsClassifier(n_neighbors=5)
score = cross_val_score(knn, x, y, cv=5, scoring='accuracy')
print(score)
k_range = range(1, 31)
k_score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
    k_score.append(scores.mean())

plt.plot(k_range, k_score)
plt.xlabel('Value of k for knn')
plt.ylabel('Cross-validated Accuracy')
plt.show()

