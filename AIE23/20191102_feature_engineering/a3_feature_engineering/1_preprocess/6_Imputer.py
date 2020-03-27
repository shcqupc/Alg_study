#coding: UTF-8
# 存在缺失值：缺失值需要补充。
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

rng = np.random.RandomState(0)

dataset = load_boston()
X_full, y_full = dataset.data, dataset.target
# print(X_full)
# print(y_full)
n_samples = X_full.shape[0]
n_features = X_full.shape[1]

# Estimate the score on the entire dataset, with no missing values
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_full, y_full, cv=5).mean()
print("Score with the entire dataset = %.2f" % score)
print(estimator)
"""
# Add missing values in 75% of the lines
missing_rate = 0.75
n_missing_samples = int(np.floor(n_samples * missing_rate))
missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                      dtype=np.bool),
                             np.ones(n_missing_samples,
                                    dtype=np.bool)))

print(missing_samples) 
rng.shuffle(missing_samples)
print(missing_samples)
missing_features = rng.randint(0, n_features, n_missing_samples)
print("missing_features")
print(missing_features)
# Estimate the score without the lines containing missing values
X_filtered = X_full[~missing_samples, :]
print(X_filtered)
y_filtered = y_full[~missing_samples]
print(y_filtered)
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_filtered, y_filtered).mean()
print("Score without the samples containing missing values = %.2f" % score)

# Estimate the score after imputation of the missing values
X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = 0
print(X_missing)
y_missing = y_full.copy()
print(y_missing)
estimator = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="mean",
                                          axis=0)),
                      ("forest", RandomForestRegressor(random_state=0,
                                                       n_estimators=100))])
score = cross_val_score(estimator, X_missing, y_missing).mean()
print("Score after imputation of the missing values = %.2f" % score)
"""