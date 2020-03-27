# 加载相关模块和库
import sys
import io
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
print(__doc__)

import pandas as pd
data_train = pd.read_csv("data/train.csv")

print("看每列统计信息", data_train.describe())

# 连续型数据

import matplotlib
import matplotlib.pyplot
rm = data_train['Age']
medv = data_train['Survived']
matplotlib.pyplot.scatter(rm, medv, c='b')
matplotlib.pyplot.show()

import matplotlib
import matplotlib.pyplot
rm = data_train['Fare']
medv = data_train['Survived']
matplotlib.pyplot.scatter(rm, medv, c='b')
matplotlib.pyplot.show()