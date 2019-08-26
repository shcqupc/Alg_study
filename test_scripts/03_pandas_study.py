import pandas as pd
import numpy as np

s = pd.Series([1, 5, 8, np.nan, 9, 10])
print(s)
dates = pd.date_range('20190826', periods=6, freq='W')
print(dates)
print(np.random.randn(6, 4))

print('\n---------------DataFrame1---------------')
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['C1', 'C2', 'C3', 'C4'])
print(df)
# print('df.dtypes', df.dtypes)
# print('df.index', df.index)
print('\n---------------select by colun---------------')
# print(df['C1'])
# print(df.C1)
print('\n---------------select by row level---------------')
# print(df[0:3])
# print(df['20190922':'20191006'])
print('\n---------------select by position---------------')
print(df.iloc[3])
print(df.iloc[3, 1])
print(df.iloc[4:6, 1:4])

print('\n---------------DataFrame2---------------')
df2 = pd.DataFrame({'c1': 1.,
                    'c2': pd.Timestamp('20190806'),
                    'c3': pd.Series(2, index=list(range(4)), dtype='float32'),
                    'c4': np.array([3] * 4, dtype='int32'),
                    'c5': pd.Categorical(['test', 'train', 'test', 'train']),
                    'c6': 'foo'})
print(df2)
print(df2.dtypes)
print('df2.size', df2.size)
