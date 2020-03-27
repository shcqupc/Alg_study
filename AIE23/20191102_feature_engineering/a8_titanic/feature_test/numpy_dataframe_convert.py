# https://blog.csdn.net/jinguangliu/article/details/78538748


from pandas import Series, DataFrame
from numpy import array
import numpy as np
import pandas as pd
arr = np.random.rand(4, 2)
array([[ 0.66867334,  0.0496808 ],
       [ 0.24225703,  0.17014163],
       [ 0.37133698,  0.3160525 ],
       [ 0.76333377,  0.54704594]])
# columns
columns_new = ['one', 'two']

# pass in array and columns
df = pd.DataFrame(arr, columns=columns_new)
print(df)
df_array = df.as_matrix()
print(df_array)