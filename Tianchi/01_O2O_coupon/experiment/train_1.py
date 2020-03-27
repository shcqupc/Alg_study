import pandas as pd
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
#####################################################
# Training
#####################################################
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, roc_auc_score

pd.set_option('display.max_columns', None)
dfoff = pickle.load(open('../data/dfoff_2.pickle', 'rb'))
dftest = pickle.load(open('../data/dftest_2.pickle', 'rb'))
print("data read end.")
# print(dftest.info())

#####################################################
# Split data
#####################################################
df = dfoff[dfoff['Label'] != -1].copy()
train = df[(df['Date_received'] < float('20160516'))].copy()
valid = df[(df['Date_received'] >= float('20160516')) & (df['Date_received'] <= float('20160615'))].copy()
# print(train['Label'].value_counts())
# print(valid['Label'].value_counts())

predictors = ['discount_rate', 'discount_type', 'discount_man', 'discount_jian', 'distance', 'weekday',
              'weekday_type', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
              'weekday_7']


#####################################################
# training
#####################################################
# check_model
def check_model(data, predictors):
    classifier = lambda: SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        max_iter=100,
        shuffle=True,
        n_jobs=1,
        class_weight=None)

    model = Pipeline(steps=[
        ('ss', StandardScaler()),
        ('en', classifier())
    ])

    parameters = {
        'en__alpha': [0.001, 0.01, 0.1],
        'en__l1_ratio': [0.001, 0.01, 0.1]
    }

    folder = StratifiedKFold(n_splits=3, shuffle=True)

    grid_search = GridSearchCV(
        model,
        parameters,
        cv=folder,
        n_jobs=-1,
        verbose=1)
    grid_search = grid_search.fit(data[predictors],
                                  data['Label'])

    return grid_search


if not os.path.isfile('../result/1_model.pkl'):
    model = check_model(train, predictors)
    print(model.best_score_)
    print(model.best_params_)
    with open('../result/1_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('../result/1_model.pkl', 'rb') as f:
        model = pickle.load(f)

y_valid_pred = model.predict_proba(valid[predictors])
# print(y_valid_pred.shape)
# print(y_valid_pred.sorted(reserve=True))
# print(y_valid_pred)
valid1 = valid.copy()
valid1['pred_prob'] = y_valid_pred[:, 1]
# print(valid1.info())

#####################################################
# avgAUC calculation
#####################################################
aucs = []
# print(valid1['Coupon_id'].value_counts())
vg = valid1.groupby(['Coupon_id'])
# vg = valid1[valid1['Coupon_id'] == 14038.0].groupby(['Coupon_id'])
for i in vg:
    # print(i)
    tmpdf = i[1].sort_values(by=["pred_prob"])
    # print(tmpdf[['Label','pred_prob']])
    if len(tmpdf['Label'].unique()) != 2:
        continue
    # print(tmpdf['Label'], tmpdf['pred_prob'])
    fpr, tpr, thresholds = roc_curve(tmpdf['Label'], tmpdf['pred_prob'], pos_label=1)
    # print(fpr, tpr, thresholds)
    aucs.append(auc(fpr, tpr))

print(np.average(aucs))

# test prediction for submission
y_test_pred = model.predict_proba(dftest[predictors])
dftest1 = dftest[['User_id', 'Coupon_id', 'Date_received']].copy()
dftest1['Label'] = y_test_pred[:, 1]
print(dftest1.info())
# dftest1.to_csv('submit1.csv', index=False, header=False)
# dftest1.head()
