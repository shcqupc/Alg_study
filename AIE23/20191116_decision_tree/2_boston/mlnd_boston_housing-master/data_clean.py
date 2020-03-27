##（1） 过滤清除异常值
# 上面我们是了解了如何取掉某个具体值，下面，我们要看看如何过滤掉某个范围的值。
# 对于数据集df，我们想过滤掉creativeID（第一列）中ID值大于10000的样本。
# [python] view plain copy
df[df['creativeID']<=10000]  

df[(df.A == 1) & (df.D == 6)]

# 计算各列数据总和并作为新列添加到末尾
df['Col_sum'] = df.apply(lambda x: x.sum(), axis=1)

# import pandas as pd

# df = pd.DataFrame({'ID':['1','2','3'], 'col_1': [0,2,3], 'col_2':[1,4,5]})
# mylist = ['a','b','c','d','e','f']

# def get_sublist(sta,end):
#     return mylist[sta:end+1]

# df['col_3'] = list(map(get_sublist,df['col_1'],df['col_2']))
# #In Python 2 don't convert above to list
# We could pass as many arguments as we wanted into the function this way. The output is what we wanted

# ID  col_1  col_2      col_3
# 0  1      0      1     [a, b]
# 1  2      2      4  [c, d, e]
# 2  3      3      5  [d, e, f]

# In [49]: df
# Out[49]: 
#           0         1
# 0  1.000000  0.000000
# 1 -0.494375  0.570994
# 2  1.000000  0.000000
# 3  1.876360 -0.229738
# 4  1.000000  0.000000

# In [50]: def f(x):    
#    ....:  return x[0] + x[1]  
#    ....:  
# axis=0，添加行
# axis=1，添加列 不同框架处理不一样
# In [51]: df.apply(f, axis=1) #passes a Series object, row-wise
# Out[51]: 
# 0    1.000000
# 1    0.076619
# 2    1.000000
# 3    1.646622
# 4    1.000000








