import pandas as pd
data_train = pd.read_csv('G:\\dl_data\\ctr\\train_sample.csv')


train_df = data_train.filter(regex='click|hour|C1|banner_pos')

print(train_df.info())

train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]

print(X[0:1])
print(y[0:1])


