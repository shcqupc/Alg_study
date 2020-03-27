import pandas as pd
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

dfoff = pickle.load(open("../data/dfoff.pickle", "rb"))
date_received = sorted(dfoff[dfoff["Date_received"].notna()]["Date_received"].unique())
date_buy = sorted(dfoff[dfoff["Date"].notna()]["Date"].unique())

couponbydate = dfoff[dfoff["Date_received"].notna()].groupby(["Date_received"])["Coupon_id"].count()
date_received_dt = pd.to_datetime(date_received, format='%Y%m%d')

buybydate = dfoff[dfoff["Date"].notna()].groupby(["Date"])["Coupon_id"].count()
date_buy_dt = pd.to_datetime(date_buy, format='%Y%m%d')

print()
# difference
# print(list(set(date_received_dt).difference(set(date_buy_dt))))  # b中有而a中没有的
# print(sorted(list(set(date_buy_dt).difference(set(date_received_dt)))))  # a中有而b中没有的

# union
# print(list(set(date_received_dt).union(set(date_buy_dt))))

# intersection
# print(list(set(date_received_dt).intersection(set(date_buy_dt))))
