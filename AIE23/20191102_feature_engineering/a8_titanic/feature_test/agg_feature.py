# -*- coding: utf-8 -*-
"""
数据预处理
如前面所说，我们的数据预处理工作占用了我们的70%时间
其完成质量直接影响最终结果
首先需要对数据有个整体的认识
"""
# 加载相关模块和库
import sys
import io
from sklearn.preprocessing import Imputer
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
print(__doc__)
import pandas as pd
df = pd.read_csv("a8_titanic/data/train.csv")
# (1) for col

# (2) for row
agg = df.Age[df.Age>2].count
#print(agg) 
# (3)
# data.groupby(['col1', 'col2'])['col3'].mean()
df2 = df["Sex"] 
temp = df.groupby(['Sex'], as_index=False)['Age'].mean()
print("agg results")
print(df.groupby(['Sex'], as_index=False)['Age'].mean())
res = pd.merge(df, temp, on=['Sex'], how='left')  # default for how='inner'
print(res[0:5])

# def f(x):
#     max_no_buy=0
#     res=[]
#     for i in x:
#         if i==0:
#             max_no_buy+=1
#             res.append(max_no_buy)
#         else:
#             max_no_buy=0
#     return 0 if len(res)==0 else max(res)
# user_nobuy= data.groupby('user_id',as_index=False)['is_trade'].agg({'user_continue_nobuy_click_cnt':lambda x:f(x)})

