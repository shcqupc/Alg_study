print(__doc__)

# The loss function used is binomial deviance. Regularization via shrinkage (learning_rate < 1.0) improves performance considerably.
# In combination with shrinkage, stochastic gradient boosting (subsample < 1.0) can produce more accurate models by reducing the variance via bagging.
# F_m(x) = F_m-1(x) + learning_rate*(当前树的预测值(也就是预测负梯度..)) //@TODO check
# learning_rate: 即每个弱学习器的权重缩减系数νν，
# 也称作步长，在原理篇的正则化章节我们也讲到了，加上了正则化项，
# 我们的强学习器的迭代公式为fk(x)=fk−1(x)+νhk(x)fk(x)=fk−1(x)+νhk(x)。νν的取值范围为0<ν≤10<ν≤1。对于同样的训练集拟合效果，较小的νν意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.datasets import load_iris, load_digits, load_boston
digits = load_digits(2)
y = digits['target']
X = digits['data']
X = X.astype(np.float32)

print("dataset len")
print(len(X))
# map labels from {-1, 1} to {0, 1}
labels, y = np.unique(y, return_inverse=True)
X_train, X_test = X[:200], X[200:]
y_train, y_test = y[:200], y[200:]

original_params = {'n_estimators': 100}
plt.figure()

for label, color, setting in [('learning_rate= 0.01', 'orange',
                               {'learning_rate': 0.01}),
                              ('learning_rate= 0.05', 'turquoise',
                               {'learning_rate': 0.05})]:
    params = dict(original_params)
    params.update(setting)

    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)

    # compute test set deviance
    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        # clf.loss_ assumes that y_test[i] in {0, 1}
        test_deviance[i] = clf.loss_(y_test, y_pred)

    plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
            '-', color=color, label=label)

plt.legend(loc='upper left')
plt.xlabel('Boosting Iterations')
plt.ylabel('Test Set Deviance')
plt.show()