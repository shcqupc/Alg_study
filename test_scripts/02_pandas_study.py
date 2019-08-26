import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dates = pd.date_range('20190826', periods=6, freq='D')
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['C1', 'C2', 'C3', 'C4'])
print('--------------------origin------------------------')
print(df)
print(' np.nan', np.nan)
df.iloc[0, 1] = np.nan
df.iloc[4, 2] = np.nan
print('\n--------------------nan------------------------')
print(df)
print('\n--------------------dropna------------------------')
print(df.dropna(axis=0, how='any'))

print('\n--------------------fillna------------------------')
print(df.fillna(value=1000))

print('\n--------------------pd.isnull------------------------')
print(pd.isnull(df))

print('\n--------------------pd.merge by column level------------------------')
df1 = pd.DataFrame(np.arange(12).reshape((6, 2)), index=dates, columns=['C1', 'C2'])
df2 = pd.DataFrame(np.arange(50, 62).reshape((6, 2)), index=dates, columns=['C3', 'C4'])
df1.index.name = 'key'
df2.index.name = 'key'
print(df1)
print('df1.index.name', df1.index.name)
print(pd.merge(df1, df2, on='key'))

print('\n--------------------pd.merge by row level------------------------')
print(np.ones(((3, 4))) * 3)
df3 = pd.DataFrame(np.ones((3, 4)) * 3, columns=['a', 'b', 'c', 'd'])
df4 = pd.DataFrame(np.ones((3, 4)) * 4, columns=['a', 'b', 'c', 'd'])
df5 = pd.DataFrame(np.ones((3, 4)) * 5, columns=['a', 'b', 'c', 'd'])
res = pd.concat([df3, df4, df5], axis=0, ignore_index=True)
print(res)

print('\n--------------------matplotlib.pyplot------------------------')
data = pd.Series(np.random.randn(1000), index=np.arange(1000))
data = data.cumsum()

data = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=['a', 'b', 'c', 'd'])
data = data.cumsum()
print(data)
ax = data.plot.scatter(x='a', y='b', color='DarkBlue', label="Class 1")
data.plot.scatter(x='a', y='c', color='LightGreen', label="Class 2", ax=ax)
plt.show()

