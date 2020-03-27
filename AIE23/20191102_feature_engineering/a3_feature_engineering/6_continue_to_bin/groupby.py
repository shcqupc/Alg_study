import pandas as pd
import numpy as np
# https://blog.csdn.net/songbinxu/article/details/79839363 
#Create a DataFrame
d = {
    'Name':['Alisa','Bobby','jodha','jack','raghu','Cathrine',
            'Alisa','Bobby','kumar','Alisa','Alex','Cathrine'],
    'Age':[26,24,23,22,23,24,26,24,22,23,24,24],
      
       'Score':[85,63,55,74,31,77,85,63,42,62,89,77]}
 
df = pd.DataFrame(d,columns=['Name','Age','Score'])
print(df.columns.values)
# key内部求和
gp = df.groupby(["Name"])["Age"].sum().reset_index() # reset_index重置index
print(gp)
gp.rename(columns={"Age":"sum_of_value"},inplace=True) # rename改列名

print(gp)

res = pd.merge(df, gp, on=['Name'], how='inner')  # default for how='inner'
print(res)
