import pandas as pd
import pickle

# offline_train = pd.read_csv('../data/ccf_offline_stage1_train.csv')
# offline_train.to_pickle('../data/offline_train.pickle')
# online_train = pd.read_csv('../data/ccf_online_stage1_train.csv')
# online_train.to_pickle('../data/online_train.pickle')
# offline_test = pd.read_csv('../data/ccf_offline_stage1_test_revised.csv')
# offline_test.to_pickle('../data/offline_test.pickle')

print('\n--------------处理offline_train----------------------')
# pkl_offline = open("../data/offline_train.pickle", "rb")
# offline_train = pickle.load(pkl_offline)

#####################################################
#  观察数据                                                 #
#####################################################
# count(*)
# print(offline_train.shape)
# print(offline_train.info(verbose=True))
# is null count(*)
# print(pd.isnull(offline_train).sum())
# 等价于 select Date_received, count(*) from offline_train group by Date_received
# print(offline_train.Date_received.value_counts(dropna=False))
# 等价于 select Distance, average(user_id), ... from offline_train group by Distance
# print(offline_train.groupby(["Distance"]).mean())
# offline_train.groupby(["Distance"])["User_id"].mean()

#####################################################
#  Feature Engineering
#####################################################
# 去掉无优惠券消费的数据
# coupon_na_idx = offline_train[offline_train.Coupon_id.isnull() == True].index
# offline_train.drop(coupon_na_idx, inplace=True)
# print(pd.isnull(offline_train).sum())
# offline_train.to_pickle('../data/offline_train.pickle')

# distance 空值填充平均值
# offline_train.Distance.fillna(value=-1, inplace=True)
# print(offline_train.Distance.value_counts(dropna=False))

# 增加 date - Date_received 列 interv
# interv = pd.to_datetime(offline_train["Date"], format="%Y%m%d") - pd.to_datetime(offline_train["Date_received"],
#                                                                                  format="%Y%m%d")
# offline_train["Interv"] = interv.dt.days


# 负样本 Date is null or interv >15
# 正样本 interv<=15
# def classify(date, interv):
#     if pd.isnull(date) or interv > 15:
#         return 0
#     else:
#         return 1

# offline_train["Y"] = offline_train.apply(lambda x: classify(x["Date"], x["Interv"]), axis=1)

# 处理完空值,丢弃 date,Date_received 和 interv 列，添加标记列Y
pkl_offline = open("../data/offline_train_1.pickle", "rb")
offline_train = pickle.load(pkl_offline)
# offline_train = offline_train.drop(["Date", "Interv", "Date_received"], axis=1)
# offline_train.to_pickle('../data/offline_train_1.pickle')

#####################################################
# 处理  Discount_rate                               #
#####################################################
print(offline_train.groupby(["Coupon_id","Y"])["User_id"].count())
print(offline_train["Date_received"])


print('\n---------------处理online_train---------------------')
# pkl_online = open("../data/online_train.pickle", "rb")
# online_train = pickle.load(pkl_online)

print('\n---------------处理offline_test---------------------')
pkl_offline_tst = open("../data/offline_test.pickle", "rb")
offline_test = pickle.load(pkl_offline_tst)
print(offline_test)
