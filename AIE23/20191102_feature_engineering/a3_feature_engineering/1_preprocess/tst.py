from sklearn.datasets import load_iris
from numpy import vstack, array, nan
import numpy as np
from sklearn.impute import SimpleImputer

iris = load_iris()
print(iris.keys())
imp1 = SimpleImputer(missing_values=np.nan,  strategy='constant', fill_value=888)
imp2 = SimpleImputer(missing_values=np.nan,  strategy='mean')
iris2 = vstack((array([nan, nan, nan, nan]), iris.data))
imp1.fit(iris2)
imp2.fit(iris2)
print('\n---------------constant---------------------')
print(imp1.transform(iris2[:5,:]))
print('\n---------------mean---------------------')
print(imp2.transform(iris2[:5,:]))





