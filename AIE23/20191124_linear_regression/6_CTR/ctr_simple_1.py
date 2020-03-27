import pandas as pd
data_train = pd.read_csv('G:\\dl_data\\ctr\\train_sample.csv')

print("看列名", data_train.columns)
print("看每列性质，空值和类型", data_train.info())
print("看每列统计信息", data_train.describe())
