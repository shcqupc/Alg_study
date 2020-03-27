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
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
print(__doc__)


import pandas as pd
data_train = pd.read_csv("data/train.csv")

print("看列名", data_train.columns)
# 数据摸底
#print("看每列性质，空值和类型", data_train.info())
# 问题1 空值填充
# 问题2 类别型数据处理

#print("看每列统计信息", data_train.describe())
# 问题1 异常值处理与归一化
# 问题2 


# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# data_train = pd.read_csv("data/train.csv")

# plt.subplot2grid((2,3),(1,0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')   # plots a kernel desnsity estimate of the subset of the 1st class passanges's age
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u"年龄")# plots an axis lable
# plt.ylabel(u"密度")
# plt.title(u"各等级的乘客年龄分布")
# plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.

# plt.show()
