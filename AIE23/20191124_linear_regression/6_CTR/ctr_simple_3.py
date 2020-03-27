import pandas as pd
from sklearn import linear_model

data_train = pd.read_csv('D:\\dl_data\\ctr\\train_sample.csv')

train_df = data_train.filter(regex='click|hour|C1|banner_pos')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]

clf = linear_model.LogisticRegression(C=100.0, penalty='l1')
clf.fit(X, y)

print(clf.predict(X[0:1]))
