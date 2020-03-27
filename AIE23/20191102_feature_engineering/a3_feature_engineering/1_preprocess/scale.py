from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np

# The function scale provides a quick and easy way to perform this operation on a single array-like dataset:
X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])
X_scaled = preprocessing.scale(X_train)
print('\n--------------scaled----------------------')
print(X_scaled)
print('\n--------------feature mean----------------------')
print(X_scaled.mean(axis=0))
print('\n--------------feature std----------------------')
print(X_scaled.std(axis=0))
"The preprocessing module further provides a utility class StandardScaler that implements the Transformer API to " \
"compute the mean and standard deviation on a training set so as to be able to later reapply the same transformation " \
"on the testing set. "
print('\n------------------StandardScaler------------------')
scaler = StandardScaler()
print(scaler.fit(X_train))
print(scaler.transform(X_train))