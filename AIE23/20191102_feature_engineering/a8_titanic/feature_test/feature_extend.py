from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
df = pd.read_csv("a8_titanic/data/train.csv")
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
train_df = df.filter(regex='Survived|Cabin|Embarked')
train_np = train_df.as_matrix()
features = train_df.columns

#(1) 离散特征扩展
f1index = df.columns.get_loc("Cabin")
f2index = df.columns.get_loc("Embarked")
print(f1index)
print(f2index)
# axis=0 apply to rows
for f1 in df["Cabin"].unique():
    for f2 in df["Embarked"].unique():
        df[str(f1)+'_'+str(f2)] = df.apply(lambda x: 1 if x[f1index] == f1 and x[f2index] == f2 else 0, axis=1)

print(df.describe())
print(df.columns)
# try to remove whose only contains 0 or 1 columns
for col in df.columns:
    print(len(df[col].unique()))
    if len(df[col].unique()) == 1:
        df.drop([col], axis=1, inplace= True)
print(df.columns)
print(df.info(max_cols=len(df.columns)))

# # (2) 原特征列运算扩展
# f1index = df.columns.get_loc("Cabin")
# f2index = df.columns.get_loc("Embarked")
# df[str(f1)+'_'+str(f2)] = df.apply(lambda x: str(x[f1index])+ str(x[f2index]), axis=1)
# print(df[str(f1)+'_'+str(f2)])

# # (3) 单列特征运算
# # 标准化某列，行view操作
# f1index = df.columns.get_loc("Age")
# sum = df["Age"].sum()
# df[str(f1)+'_'+str(f2)] = df.apply(lambda x: x[f1index] / sum, axis=1)
# print(df[str(f1)+'_'+str(f2)])
# # 列的apply没太大意义
# df[str(f1)+'_'+str(f2)] = df["Age"] / sum