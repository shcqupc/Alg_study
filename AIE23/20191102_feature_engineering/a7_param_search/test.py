from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
iris = load_iris()
X = iris.data
y = iris.target
print(X[0:5])
print(y[0:5])
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
result = np.sum(pow((y_pred - y), 2))
print(result)
print(y_pred[0:5])
columns_new = ['sepal length', 'sepal width', 'petal length', 'petal width']
# pass in array and columns
df = pd.DataFrame(X, columns=columns_new)
print(df.info())
# columns
#1. sepal length in cm
#2. sepal width in cm
#3. petal length in cm
#4. petal width in cm
import matplotlib.pyplot as plt
print(X[0])
#plt.scatter(X[0:len(X)][0], y)
#plt.show()
