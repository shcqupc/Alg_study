import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ds = pd.read_csv("../0_DT_Tree/data/DS_Adaboost.csv")
ds.Decision = np.where(ds.Decision == 1, 1, -1)


# print(ds)


def findDecision(x1, x2):
    if x1 > 2.1: return -0.025  # calculated by Regression Tree
    if x1 <= 2.1: return 0.1


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


print('\n------------------------------------')
# step1
ds['weight'] = ds.apply(lambda x: 0.1, axis=1)
# step2
ds['weighted_actual'] = ds.apply(lambda x: x.Decision / 10, axis=1)
# to use weighted actual as target value whereas x1 and x2 are features to build a decision stump
# refer to regression decision tree
ds['prediction'] = ds.apply(lambda x: sign(findDecision(x.x1, x.x2)), axis=1)
# step3 get actual loss
ds['loss'] = ds.apply(lambda x: 0 if x.prediction == x.Decision else 1, axis=1)
ds['wxl'] = ds.apply(lambda x: x.weight * x.loss, axis=1)

epsilon = ds["wxl"].sum()  # get total error by sum(wxl)=0.3
alpha1 = 1 / 2 * np.log((1 - epsilon) / epsilon)
print('epsilon:', epsilon, 'alpha1:', alpha1)
ds["wi"] = ds.apply(lambda x: x.weight * np.exp(-1 * alpha1 * x.Decision * x.prediction), axis=1)
ds["norm_w"] = ds["wi"] / ds["wi"].sum()  # Normalize wi
# w_array = np.array(ds["wi"].values).reshape(-1,1)
# ds["norm_w"] = StandardScaler().fit_transform(w_array)
# print(ds)
pos = ds[ds["Decision"] >= 0]
neg = ds[ds["Decision"] < 0]
plt.figure(figsize=[5, 3.5])
plt.scatter(pos['x1'], pos['x2'], marker='+', s=500 * abs(pos['weighted_actual']), c='blue')
plt.scatter(neg['x1'], neg['x2'], marker='_', s=500 * abs(neg['weighted_actual']), c='red')
# plt.show()

print('\n------------------------------------')
#####################################################
# Round 2
#####################################################
# step1 shift normalized w_(i+1) column i.e.norm_w to weight column
ds.weight = ds.norm_w
# cols = [4, 5, 6, 7, 8, 9]
# ds = ds.drop(ds.columns[cols], axis=1, inplace=False)
# get weighted_actual by current weight
ds['weighted_actual2'] = ds.apply(lambda x: x.Decision * x.weight, axis=1)


def findDecision2(x1, x2):
    if x2 <= 3.5:
        return -0.02380952380952381
    if x2 > 3.5:
        return 0.10714285714285714


ds['prediction2'] = ds.apply(lambda x: sign(findDecision2(x.x1, x.x2)), axis=1)
ds['loss2'] = ds.apply(lambda x: 0 if x.prediction2 == x.Decision else 1, axis=1)
ds['wxl2'] = ds.apply(lambda x: x.weight * x.loss2, axis=1)

epsilon = ds["wxl2"].sum()  # get total error by sum(wxl)=0.3
alpha2 = 1 / 2 * np.log((1 - epsilon) / epsilon)
print('epsilon2:', epsilon, 'alpha2:', alpha2)
ds["wi2"] = ds.apply(lambda x: x.weight * np.exp(-1 * alpha2 * x.Decision * x.prediction2), axis=1)
ds["norm_w2"] = ds["wi2"] / ds["wi2"].sum()  # Normalize wi
# print(ds.iloc[:, 6:])
# ds.to_csv('adaboost_round2.csv')

pos = ds[ds["Decision"] >= 0]
neg = ds[ds["Decision"] < 0]
# plt.figure(figsize=[5, 3.5])
plt.scatter(pos['x1'], pos['x2'], marker='+', s=500 * abs(pos['weighted_actual2']), c='blue')
plt.scatter(neg['x1'], neg['x2'], marker='_', s=500 * abs(neg['weighted_actual2']), c='red')
# plt.show()
print('\n------------------------------------')
#####################################################
# Round 3
#####################################################
# step1 shift normalized w_(i+1) column i.e.norm_w to weight column
ds.weight = ds.norm_w2
# get weighted_actual by current weight
ds['weighted_actual3'] = ds.apply(lambda x: x.Decision * x.weight, axis=1)


def findDecision3(x1, x2):
    if x1 > 2.1:
        return -0.003787878787878794
    if x1 <= 2.1:
        return 0.16666666666666666


ds['prediction3'] = ds.apply(lambda x: sign(findDecision3(x.x1, x.x2)), axis=1)
ds['loss3'] = ds.apply(lambda x: 0 if x.prediction3 == x.Decision else 1, axis=1)
ds['wxl3'] = ds.apply(lambda x: x.weight * x.loss3, axis=1)

epsilon = ds["wxl3"].sum()  # get total error by sum(wxl)=0.3
alpha3 = 1 / 2 * np.log((1 - epsilon) / epsilon)
print('epsilon3:', epsilon, 'alpha3:', alpha3)
ds["wi3"] = ds.apply(lambda x: x.weight * np.exp(-1 * alpha3 * x.Decision * x.prediction3), axis=1)
ds["norm_w3"] = ds["wi3"] / ds["wi3"].sum()  # Normalize wi
# ds.to_csv('adaboost_round3.csv')

pos = ds[ds["Decision"] >= 0]
neg = ds[ds["Decision"] < 0]
# plt.figure(figsize=[5, 3.5])
plt.scatter(pos['x1'], pos['x2'], marker='+', s=500 * abs(pos['weighted_actual2']), c='blue')
plt.scatter(neg['x1'], neg['x2'], marker='_', s=500 * abs(neg['weighted_actual2']), c='red')
# plt.show()


print('\n------------------------------------')
#####################################################
# Round 4
#####################################################
# step1 shift normalized w_(i+1) column i.e.norm_w to weight column
ds.weight = ds.norm_w3
# get weighted_actual by current weight
ds['weighted_actual4'] = ds.apply(lambda x: x.Decision * x.weight, axis=1)


def findDecision4(x1, x2):
    if x1 <= 6.0: return 0.08055555555555555
    if x1 > 6.0: return -0.07777777777777778


ds['prediction4'] = ds.apply(lambda x: sign(findDecision4(x.x1, x.x2)), axis=1)
ds['loss4'] = ds.apply(lambda x: 0 if x.prediction4 == x.Decision else 1, axis=1)
ds['wxl4'] = ds.apply(lambda x: x.weight * x.loss4, axis=1)

epsilon = ds["wxl4"].sum()  # get total error by sum(wxl)=0.3
alpha4 = 1 / 2 * np.log((1 - epsilon) / epsilon)
print('epsilon4:', epsilon, 'alpha4:', alpha4)
ds["wi4"] = ds.apply(lambda x: x.weight * np.exp(-1 * alpha4 * x.Decision * x.prediction4), axis=1)
ds["norm_w4"] = ds["wi4"] / ds["wi4"].sum()  # Normalize wi
# ds.to_csv('adaboost_round4.csv')

pos = ds[ds["Decision"] >= 0]
neg = ds[ds["Decision"] < 0]
# plt.figure(figsize=[5, 3.5])
plt.scatter(pos['x1'], pos['x2'], marker='+', s=500 * abs(pos['weighted_actual4']), c='blue')
plt.scatter(neg['x1'], neg['x2'], marker='_', s=500 * abs(neg['weighted_actual4']), c='red')
plt.show()

#####################################################
#  Cumulative sum of each round’s alpha times prediction gives the final prediction
#####################################################
ds["PREDICTION"] = ds.apply(
    lambda x: sign(alpha1 * x.prediction + alpha2 * x.prediction2 + alpha3 * x.prediction3 + alpha4 * x.prediction4), axis=1)
ds.to_csv('adaboost_round5.csv')

#####################################################
#   Pruning
#####################################################
