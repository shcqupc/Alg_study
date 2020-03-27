import pandas as pd
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

pd.set_option('display.max_columns', None)
# pd.read_csv("../data/ccf_offline_stage1_train.csv").to_pickle('../data/dfoff.pickle')
# pd.read_csv("../data/ccf_offline_stage1_test_revised.csv").to_pickle('../data/dftest.pickle')
# pd.read_csv("../data/ccf_online_stage1_train.csv").to_pickle('../data/dfon.pickle')

# dfoff = pickle.load(open("../data/dfoff.pickle", "rb"))
# dftest = pickle.load(open("../data/dftest.pickle", "rb"))
# dfon = pickle.load(open("../data/dfon.pickle", "rb"))
# print("data read end.")


# 观察特征类型
# print(dfoff.info())
# 观察空值
# print(pd.isnull(dfoff).sum(axis=0))
#####################################################
# User_id               0
# Merchant_id           0
# Coupon_id        701602
# Discount_rate    701602
# Distance         106003
# Date_received    701602
# Date             977900
#####################################################
# 观察Discount_rate
# print(dfoff["Discount_rate"].value_counts(dropna=False))
'''
# 观察Date_received和Date
# print(dfoff["Date_received"].value_counts(dropna=False))
date_received = sorted(dfoff[dfoff["Date_received"].notna()]["Date_received"].unique())
date_buy = sorted(dfoff[dfoff["Date"].notna() & dfoff["Date_received"].notna()]["Date"].unique())
print('date_received', date_received[0], date_received[-1])
print('date_buy', date_buy[0], date_buy[-1])

# print(list(set(date_received).difference(set(date_buy))))
# print(list(set(date_buy).difference(set(date_received))))

# couponbydate = dfoff[dfoff["Date_received"].notna()].groupby(["Date_received"]).count()
couponbydate = dfoff["Date_received"].value_counts(dropna=True)
couponbydate = couponbydate.sort_index()
date_received_dt = pd.to_datetime(date_received, format='%Y%m%d')

# buybydate = dfoff[(dfoff['Date'] != 'null') & (dfoff['Date_received'] != 'null')].groupby(["Date_received"], as_index=False).count()
buybydate = dfoff[dfoff["Date"] < 20160616]["Date"].value_counts(dropna=True)
buybydate = buybydate.sort_index()
print(couponbydate)
print(buybydate)
date_buy_dt = pd.to_datetime(date_received, format='%Y%m%d')

plt.subplot(211)
plt.bar(date_received_dt, couponbydate.values, label='number of coupon received')
plt.bar(date_buy_dt, buybydate.values, label='number of coupon used')
plt.yscale('log')
plt.ylabel('Count')
plt.legend()

plt.subplot(212)
plt.bar(date_received_dt, buybydate.values / couponbydate.values)
plt.ylabel('Ratio(coupon used/coupon received)')


plt.show() 
'''
'''
date_received = dfoff[dfoff["Date_received"].notna()]["Date_received"].unique()
date_received = sorted(date_received)
date_buy = dfoff[dfoff["Date"].notna()]["Date"].unique()
date_buy = sorted(date_buy)
print('优惠券收到日期从',date_received[0],'到', date_received[-1])
print('消费日期从', date_buy[0], '到', date_buy[-1])
couponbydate = dfoff[dfoff['Date_received'] != 'null'][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
couponbydate.columns = ['Date_received','count']
buybydate = dfoff[(dfoff['Date'] != 'null') & (dfoff['Date_received'] != 'null')][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
buybydate.columns = ['Date_received','count']

# plt.figure(figsize = (12,8))
date_received_dt = pd.to_datetime(date_received, format='%Y%m%d')

plt.subplot(211)
plt.bar(date_received_dt, couponbydate['count'], label = 'number of coupon received' )
plt.bar(date_received_dt, buybydate['count'], label = 'number of coupon used')
plt.yscale('log')
plt.ylabel('Count')
plt.legend()

plt.subplot(212)
plt.bar(date_received_dt, buybydate['count']/couponbydate['count'])
plt.ylabel('Ratio(coupon used/coupon received)')
plt.tight_layout()
plt.show()
'''


#####################################################
# tools
#####################################################
# 把满减转换成折扣率
def convertRate(discount):
    if ":" in discount:
        sp = discount.split(":")
        return 1 - float(sp[1]) / float(sp[0])
    elif discount == 'nan':
        return 1
    else:
        return discount


# 获取满减的满值
def getDiscountMan(discount):
    if ':' in discount:
        sp = discount.split(':')
        return int(sp[0])
    else:
        return 0


# 获取满减的减值
def getDiscountJian(discount):
    if ':' in discount:
        sp = discount.split(':')
        return int(sp[1])
    else:
        return 0


# 获取打折类型 空值返回np.nan, 满减返回 1， 否则返回0
def getDiscountType(discount):
    if pd.isnull(discount):
        return np.nan
    elif ":" in discount:
        return 1
    else:
        return 0


#####################################################
# data processing
#####################################################
def processData(df):
    df["discount_rate"] = df.apply(lambda x: convertRate(str(x["Discount_rate"])), axis=1)
    df["discount_man"] = df.apply(lambda x: getDiscountMan(str(x["Discount_rate"])), axis=1)
    df["discount_jian"] = df.apply(lambda x: getDiscountJian(str(x["Discount_rate"])), axis=1)
    df["discount_type"] = df.apply(lambda x: getDiscountType(str(x["Discount_rate"])), axis=1)
    df["distance"] = df["Distance"].fillna(-1).astype(int)
    print("end process data")
    return df


# dfoff = pickle.load(open("../data/dfoff.pickle", "rb"))
# dftest = pickle.load(open("../data/dftest.pickle", "rb"))
# print("data read end.")
# processData(dfoff).to_pickle('../data/dfoff_1.pickle')
# processData(dftest).to_pickle('../data/dftest_1.pickle')

#####################################################
# 处理date和date_received
#####################################################
# dfoff = pickle.load(open('../data/dfoff_1.pickle', 'rb'))
# dftest = pickle.load(open('../data/dftest_1.pickle','rb'))
# print("data read end.")

# 等价于 select Date,Date_received,count(*) Count from dfoff where Date is not null and Date_received is not null
# gb1 = dfoff.groupby(["Date", "Date_received"])["User_id"].count().sort_values(ascending=False).reset_index(name="Count")
# pd.set_option('display.max_rows', None
# print(type(gb1), gb1.head(100))


# 将日期转换成星期，星期一：1
def getWeekday(row):
    if row == 'nan':
        return np.nan
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1


# 获取从领券到消费的时间周期
def getLabel(row):
    if pd.isnull(row['Date_received']):
        return -1
    if pd.notnull(row['Date']):
        days = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'],
                                                                             format='%Y%m%d')
        if days <= pd.Timedelta(15, 'D'):
            return 1
    return 0


def extendDate(df):
    df['weekday'] = df['Date_received'].astype(str).apply(lambda x: getWeekday(x))
    df['weekday_type'] = df['weekday'].apply(lambda x: 1 if x in (6, 7) else 0)
    weekdaycols = ['weekday_' + str(i) for i in range(1, 8)]
    df[weekdaycols] = pd.get_dummies(df['weekday'])
    return df


#
# dfoff = extendDate(dfoff)
# dfoff['Label'] = dfoff.apply(lambda x: getLabel(x), axis=1)
# dfoff.to_pickle('../data/dfoff_2.pickle')
# print('end dfoff_2.pickle')
# extendDate(dftest).to_pickle('../data/dftest_2.pickle')
# print('end dftest_2.pickle')

dfoff = pickle.load(open('../data/dfoff_2.pickle', 'rb'))
dftest = pickle.load(open('../data/dftest_2.pickle', 'rb'))
print("data read end.")
print(dfoff.info())
print(dftest.info())
print(dfoff[(dfoff['Date_received'] < float('20160516'))].count())
print(dfoff["Label"].value_counts(dropna=False))
print(dfoff["Distance"].value_counts(dropna=False))
