import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris, load_digits, load_boston
# 如果迭代次数过多，也会进入过拟合。表现就是随着迭代次数的增加，测试集上的测试误差开始下降。
# 当开始过拟合或者过训练时，测试集上的测试误差开始上升，或者说波动。
# 说明：设置early_stopping_rounds=10，当loss在10轮迭代之内，都没有提升的话，就stop。如果说eval_metric有很多个指标，那就以最后一个指标为准。
# 当迭代次数过多时，测试集上的测试误差基本上已经不再下降。并且测试误差基本上已经在一个水平附近波动，甚至下降。说明，已经进入了过训练阶段
#Early stopping is an approach to training complex machine learning models to avoid overfitting.
#It works by monitoring the performance of the model that is being trained on a separate test 
# dataset and stopping the training procedure once the performance on the test dataset has not improved after a fixed number of training iterations.
rng = np.random.RandomState(31337)

print("Zeros and Ones from the Digits dataset: binary classification")
digits = load_digits(2)

# Early-stopping

X = digits['data']
y = digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train, early_stopping_rounds=3, eval_metric="auc",
        eval_set=[(X_test, y_test)])
