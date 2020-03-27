import numpy as np


# import sklearn.datasets as sd
# print(np.random.random())
#
# iris = sd.load_iris()
# print(iris.keys())
# x = iris['feature_names']
# y = iris.target
# print(x)


def steuDistance(instance1, instance2, length):
    tst_instance = []
    tra_instance = []
    for x in range(length):
        tst_instance.append(float(instance1[x]))
        tra_instance.append(float(instance2[x]))
    print(tst_instance)
    print(tra_instance)
    X = np.vstack([tst_instance, tra_instance])
    print('vstack', X)
    var = np.var(X, axis=0, ddof=1)
    print('var',var)
    return np.sqrt(((np.array(tst_instance) - np.array(tra_instance)) ** 2 / var).sum())


# l1 = [5.6, 2.8, 4.9, 2.0, 'Iris - virginica']
# l2 = [5.5, 2.3, 4.0, 1.3, 'Iris - versicolor']
# l1 = np.array([5.6, 2.8, 4.9, 2.0, 'Iris - virginica'])
# l2 = np.array([5.5, 2.3, 4.0, 1.3, 'Iris - versicolor'])
l1 = [6.2, 3.4, 5.4, 2.3]
l2 = [6.8, 3.2, 5.9, 2.3]

print(np.array(l1) - np.array(l2))
print((np.array(l1) - np.array(l2))**2)
print(steuDistance(l1, l2, 4))

print(np.isnan(np.nan))
