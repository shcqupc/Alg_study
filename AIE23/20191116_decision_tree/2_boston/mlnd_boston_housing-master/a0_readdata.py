# 载入此项目所需要的库
import numpy as np
import pandas as pd
import visuals as vs # Supplementary code

# 载入波士顿房屋的数据集
data = pd.read_csv('2_boston/mlnd_boston_housing-master/housing.csv')

print(data.info())
print(data.describe())