print(__doc__)
import pandas as pd
data_train = pd.read_csv("a8_titanic/data/train.csv")
print("看列名", data_train.columns)
# 数据摸底
print("看每列性质，空值和类型", data_train.info())
#print(data_train[0:10])
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
print(dummies_Cabin.info())
print(dummies_Cabin[0:3])