import pandas as pd
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
dfoff = pickle.load(open("../data/dfoff_2.pickle", "rb"))
# dftest = pickle.load(open("../data/dftest_2.pickle", "rb"))
# dfon = pickle.load(open("../data/dfon.pickle", "rb"))
print("data read end.")

#####################################################
# Feature Selection
#####################################################
# print(dfoff['Date'].value_counts(dropna=False))
feature = dfoff[(dfoff['Date'] < 20160516) | (dfoff['Date'].isna() & (dfoff['Date_received'] < 20160516))].copy()
data = dfoff[(dfoff['Date_received'] >= 20160516) & (dfoff['Date_received'] <= 20160615)].copy()
fdf = feature.copy()

# key of user
u = fdf[['User_id']].copy().drop_duplicates()

# u_coupon_count : num of coupon received by user
u1 = fdf[fdf['Date_received'].notna()][['User_id']].copy()
u1['u_coupon_count'] = 1
u1 = u1.groupby(['User_id'], as_index=False).count()
# print(u1.head(2))

# u_buy_count : times of user buy offline (with or without coupon)
u2 = fdf[fdf['Date'].notna()][['User_id']].copy()
u2['u_buy_count'] = 1
u2 = u2.groupby(['User_id'], as_index=False).count()
# print(u2.head(2))

# u_buy_with_coupon : times of user buy offline (with coupon)
u3 = fdf[fdf['Date'].notna() & fdf['Date_received'].notna()][['User_id']].copy()
u3['u_buy_with_coupon'] = 1
u3 = u3.groupby(['User_id'], as_index=False).count()
# print(u3.head(2))

# u_merchant_count : num of merchant user bought from
u4 = fdf[fdf['Date'].notna()][['User_id', 'Merchant_id']].copy().drop_duplicates()
u4 = u4.groupby(['User_id'], as_index=False).count()
u4.rename(columns={'Merchant_id': 'u_buy_merchant_count'}, inplace=True)
# print(u4.head(4))

# u_min_distance
utmp = fdf[(fdf['Date'].notna()) & (fdf['Date_received'].notna())][['User_id', 'distance']].copy()
utmp.replace(-1, np.nan, inplace=True)
u5 = utmp.groupby(['User_id'], as_index=False).min()
u5.rename(columns={'distance': 'u_min_distance'}, inplace=True)
u6 = utmp.groupby(['User_id'], as_index=False).max()
u6.rename(columns={'distance': 'u_max_distance'}, inplace=True)
u7 = utmp.groupby(['User_id'], as_index=False).mean()
u7.rename(columns={'distance': 'u_mean_distance'}, inplace=True)
u8 = utmp.groupby(['User_id'], as_index=False).median()
u8.rename(columns={'distance': 'u_median_distance'}, inplace=True)
# print(u.shape, u1.shape, u2.shape, u3.shape, u4.shape, u5.shape, u6.shape, u7.shape, u8.shape)

# merge all the features on key User_id
user_feature = pd.merge(u, u1, on='User_id', how='left')
user_feature = pd.merge(user_feature, u2, on='User_id', how='left')
user_feature = pd.merge(user_feature, u3, on='User_id', how='left')
user_feature = pd.merge(user_feature, u4, on='User_id', how='left')
user_feature = pd.merge(user_feature, u5, on='User_id', how='left')
user_feature = pd.merge(user_feature, u6, on='User_id', how='left')
user_feature = pd.merge(user_feature, u7, on='User_id', how='left')
user_feature = pd.merge(user_feature, u8, on='User_id', how='left')
# print(user_feature)

# calculate rate
user_feature['u_use_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float') / user_feature[
    'u_coupon_count'].astype('float')
user_feature['u_buy_with_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float') / user_feature[
    'u_buy_count'].astype('float')
user_feature = user_feature.fillna(0)
# print(user_feature[user_feature['u_buy_with_coupon_rate'] != 0].head(2))

# add user feature to data on key User_id
data2 = pd.merge(data, user_feature, on='User_id', how='left').fillna(0)

# split data2 into valid and train
train, valid = train_test_split(data2, test_size=0.2, stratify=data2['Label'], random_state=100)
print(train.shape, valid.shape)
