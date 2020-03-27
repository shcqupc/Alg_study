import pandas as pd
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from sklearn.model_selection import train_test_split
from o2o_fetools import featureProcess
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, roc_auc_score


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


pd.set_option('display.max_columns', None)
# dfoff = pickle.load(open("../data/dfoff_2.pickle", "rb"))
# dftest = pickle.load(open("../data/dftest_2.pickle", "rb"))
# dfon = pickle.load(open("../data/dfon.pickle", "rb"))
# print("data read end.")

# feature = dfoff[(dfoff['Date'] < 20160516) | (dfoff['Date'].isna() & (dfoff['Date_received'] < 20160516))].copy()
# data = dfoff[(dfoff['Date_received'] >= 20160516) & (dfoff['Date_received'] <= 20160615)].copy()


# feature engineering
# train, test = featureProcess(dfoff, dfoff, dftest)
# train, test = featureProcess(feature, data, dftest)
train = pickle.load(open('../data/train.pickle', "rb"))
test = pickle.load(open('../data/test.pickle', "rb"))

#####################################################
# check trainning data
#####################################################
# user_feature = pickle.load(open('../data/user_feature.pickle', "rb"))
# merchant_feature = pickle.load(open('../data/merchant_feature.pickle', "rb"))
# user_merchant_feature = pickle.load(open('../data/user_merchant_feature.pickle', "rb"))


# select count(*) from feature a, data b where a.User_id=b.User_id
# comp1 = pd.merge(feature, data, on='User_id')
# print(feature.shape, data.shape, comp1.shape, user_feature.shape)
# print(feature['User_id'].nunique(), data['User_id'].nunique(), comp1['User_id'].nunique(),
#       user_feature['User_id'].nunique())

# select count(*) from feature a, data b where a.Merchant_id=b.Merchant_id group by a.Merchant_id
# select count(*) from feature a, data b where a.User_id=b.User_id and a.Merchant_id=b.Merchant_id group by a.User_id,Merchant_id


# features
predictors = ['discount_rate', 'discount_man', 'discount_jian', 'discount_type', 'distance',
              'weekday', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
              'weekday_7', 'weekday_type',
              'u_coupon_count', 'u_buy_count', 'u_buy_with_coupon', 'u_merchant_count', 'u_min_distance',
              'u_max_distance', 'u_mean_distance', 'u_median_distance', 'u_use_coupon_rate', 'u_buy_with_coupon_rate',
              'm_coupon_count', 'm_sale_count', 'm_sale_with_coupon', 'm_min_distance', 'm_max_distance',
              'm_mean_distance', 'm_median_distance', 'm_coupon_use_rate', 'm_sale_with_coupon_rate', 'um_count',
              'um_buy_count',
              'um_coupon_count', 'um_buy_with_coupon', 'um_buy_rate', 'um_coupon_use_rate', 'um_buy_with_coupon_rate']

trainSub, validSub = train_test_split(train, test_size=0.2, stratify=train['Label'], random_state=100)
# print(trainSub['discount_rate'].unique())
# print(feature['discount_rate'].unique())


# model = check_model(trainSub, predictors)
# if not os.path.isfile('../result/2_model.pkl'):
#     model = check_model(train, predictors)
#     print(model.best_score_)
#     print(model.best_params_)
#     with open('../result/2_model.pkl', 'wb') as f:
#         pickle.dump(model, f)
# else:
#     with open('../result/2_model.pkl', 'rb') as f:
#         model = pickle.load(f)

import xgboost as xgb

print("Pickling sklearn API models")
if not os.path.isfile('../result/3_xgb_model.pkl'):
    xgb_model = xgb.XGBRegressor()
    model = GridSearchCV(xgb_model,
                       {'max_depth': [2, 4, 6],
                        'n_estimators': [50, 100, 200]}, verbose=1)
    model.fit(trainSub[predictors], trainSub['Label'])
    print(model.best_score_)
    print(model.best_params_)
    with open('../result/3_xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('../result/3_xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)

validSub.loc[:, 'pred_prob'] = model.predict(validSub[predictors])
print(min(validSub['pred_prob']))
# validSub.loc[:, 'pred_prob'] = model.predict_proba(validSub[predictors])[:, 1]
validgroup = validSub.groupby(['Coupon_id'])
aucs = []
for i in validgroup:
    tmpdf = i[1]
    if len(tmpdf['Label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['Label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))
