import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from sklearn.model_selection import train_test_split


def userFeature(df):
    u = df[['User_id']].copy().drop_duplicates()

    # u_coupon_count : num of coupon received by user
    u1 = df[df['Date_received'].notna()][['User_id']].copy()
    u1['u_coupon_count'] = 1
    u1 = u1.groupby(['User_id'], as_index=False).count()

    # u_buy_count : times of user buy offline (with or without coupon)
    u2 = df[df['Date'].notna()][['User_id']].copy()
    u2['u_buy_count'] = 1
    u2 = u2.groupby(['User_id'], as_index=False).count()

    # u_buy_with_coupon : times of user buy offline (with coupon)
    u3 = df[((df['Date'].notna()) & (df['Date_received'].notna()))][['User_id']].copy()
    u3['u_buy_with_coupon'] = 1
    u3 = u3.groupby(['User_id'], as_index=False).count()

    # u_merchant_count : num of merchant user bought from
    u4 = df[df['Date'].notna()][['User_id', 'Merchant_id']].copy()
    u4.drop_duplicates(inplace=True)
    u4 = u4.groupby(['User_id'], as_index=False).count()
    u4.rename(columns={'Merchant_id': 'u_merchant_count'}, inplace=True)

    # u_min_distance
    utmp = df[(df['Date'].notna()) & (df['Date_received'].notna())][['User_id', 'distance']].copy()
    utmp.replace(-1, np.nan, inplace=True)
    u5 = utmp.groupby(['User_id'], as_index=False).min()
    u5.rename(columns={'distance': 'u_min_distance'}, inplace=True)
    u6 = utmp.groupby(['User_id'], as_index=False).max()
    u6.rename(columns={'distance': 'u_max_distance'}, inplace=True)
    u7 = utmp.groupby(['User_id'], as_index=False).mean()
    u7.rename(columns={'distance': 'u_mean_distance'}, inplace=True)
    u8 = utmp.groupby(['User_id'], as_index=False).median()
    u8.rename(columns={'distance': 'u_median_distance'}, inplace=True)

    user_feature = pd.merge(u, u1, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u2, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u3, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u4, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u5, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u6, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u7, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u8, on='User_id', how='left')

    user_feature['u_use_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float') / user_feature[
        'u_coupon_count'].astype('float')
    user_feature['u_buy_with_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float') / user_feature[
        'u_buy_count'].astype('float')
    user_feature = user_feature.fillna(0)

    # print(user_feature.columns.tolist())
    return user_feature


def merchantFeature(df):
    m = df[['Merchant_id']].copy().drop_duplicates()

    # m_coupon_count : num of coupon from merchant
    m1 = df[df['Date_received'].notna()][['Merchant_id']].copy()
    m1['m_coupon_count'] = 1
    m1 = m1.groupby(['Merchant_id'], as_index=False).count()

    # m_sale_count : num of sale from merchant (with or without coupon)
    m2 = df[df['Date'].notna()][['Merchant_id']].copy()
    m2['m_sale_count'] = 1
    m2 = m2.groupby(['Merchant_id'], as_index=False).count()

    # m_sale_with_coupon : num of sale from merchant with coupon usage
    m3 = df[(df['Date'].notna()) & (df['Date_received'].notna())][['Merchant_id']].copy()
    m3['m_sale_with_coupon'] = 1
    m3 = m3.groupby(['Merchant_id'], as_index=False).count()

    # m_min_distance
    mtmp = df[(df['Date'].notna()) & (df['Date_received'].notna())][['Merchant_id', 'distance']].copy()
    mtmp.replace(-1, np.nan, inplace=True)
    m4 = mtmp.groupby(['Merchant_id'], as_index=False).min()
    m4.rename(columns={'distance': 'm_min_distance'}, inplace=True)
    m5 = mtmp.groupby(['Merchant_id'], as_index=False).max()
    m5.rename(columns={'distance': 'm_max_distance'}, inplace=True)
    m6 = mtmp.groupby(['Merchant_id'], as_index=False).mean()
    m6.rename(columns={'distance': 'm_mean_distance'}, inplace=True)
    m7 = mtmp.groupby(['Merchant_id'], as_index=False).median()
    m7.rename(columns={'distance': 'm_median_distance'}, inplace=True)

    merchant_feature = pd.merge(m, m1, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m2, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m3, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m4, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m5, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m6, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m7, on='Merchant_id', how='left')

    merchant_feature['m_coupon_use_rate'] = merchant_feature['m_sale_with_coupon'].astype('float') / merchant_feature[
        'm_coupon_count'].astype('float')
    merchant_feature['m_sale_with_coupon_rate'] = merchant_feature['m_sale_with_coupon'].astype('float') / \
                                                  merchant_feature['m_sale_count'].astype('float')
    merchant_feature = merchant_feature.fillna(0)

    # print(merchant_feature.columns.tolist())
    return merchant_feature


def usermerchantFeature(df):
    um = df[['User_id', 'Merchant_id']].copy().drop_duplicates()

    um1 = df[['User_id', 'Merchant_id']].copy()
    um1['um_count'] = 1
    um1 = um1.groupby(['User_id', 'Merchant_id'], as_index=False).count()

    um2 = df[df['Date'].notna()][['User_id', 'Merchant_id']].copy()
    um2['um_buy_count'] = 1
    um2 = um2.groupby(['User_id', 'Merchant_id'], as_index=False).count()

    um3 = df[df['Date_received'].notna()][['User_id', 'Merchant_id']].copy()
    um3['um_coupon_count'] = 1
    um3 = um3.groupby(['User_id', 'Merchant_id'], as_index=False).count()

    um4 = df[(df['Date_received'].notna()) & (df['Date'].notna())][['User_id', 'Merchant_id']].copy()
    um4['um_buy_with_coupon'] = 1
    um4 = um4.groupby(['User_id', 'Merchant_id'], as_index=False).count()

    user_merchant_feature = pd.merge(um, um1, on=['User_id', 'Merchant_id'], how='left')
    user_merchant_feature = pd.merge(user_merchant_feature, um2, on=['User_id', 'Merchant_id'], how='left')
    user_merchant_feature = pd.merge(user_merchant_feature, um3, on=['User_id', 'Merchant_id'], how='left')
    user_merchant_feature = pd.merge(user_merchant_feature, um4, on=['User_id', 'Merchant_id'], how='left')
    user_merchant_feature = user_merchant_feature.fillna(0)

    user_merchant_feature['um_buy_rate'] = user_merchant_feature['um_buy_count'].astype('float') / \
                                           user_merchant_feature['um_count'].astype('float')
    user_merchant_feature['um_coupon_use_rate'] = user_merchant_feature['um_buy_with_coupon'].astype('float') / \
                                                  user_merchant_feature['um_coupon_count'].astype('float')
    user_merchant_feature['um_buy_with_coupon_rate'] = user_merchant_feature['um_buy_with_coupon'].astype('float') / \
                                                       user_merchant_feature['um_buy_count'].astype('float')
    user_merchant_feature = user_merchant_feature.fillna(0)

    # print(user_merchant_feature.columns.tolist())
    return user_merchant_feature


def featureProcess(feature, train, test):
    """
    feature engineering from feature data
    then assign user, merchant, and user_merchant feature for train and test
    """

    user_feature = userFeature(feature)
    merchant_feature = merchantFeature(feature)
    user_merchant_feature = usermerchantFeature(feature)
    user_feature.to_pickle('../data/user_feature.pickle')
    merchant_feature.to_pickle('../data/merchant_feature.pickle')
    user_merchant_feature.to_pickle('../data/user_merchant_feature.pickle')

    train = pd.merge(train, user_feature, on='User_id', how='left')
    train = pd.merge(train, merchant_feature, on='Merchant_id', how='left')
    train = pd.merge(train, user_merchant_feature, on=['User_id', 'Merchant_id'], how='left')
    train = train.fillna(0)

    test = pd.merge(test, user_feature, on='User_id', how='left')
    test = pd.merge(test, merchant_feature, on='Merchant_id', how='left')
    test = pd.merge(test, user_merchant_feature, on=['User_id', 'Merchant_id'], how='left')
    test = test.fillna(0)

    train.to_pickle('../data/train.pickle')
    test.to_pickle('../data/test.pickle')
    return train, test