import pandas as pd
data_train = pd.read_csv("a8_titanic/data/train.csv")
print(data_train.columns)
print(data_train.info())