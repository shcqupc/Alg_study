import numpy as np
import pandas as pd
n_samples = 10
a = np.random.randint(0, 10, n_samples)

# say you want to split at 1 and 3
boundaries = [1, 3]
# add min and max values of your data
boundaries = sorted({a.min(), a.max() + 1} | set(boundaries))

a_discretized_1 = pd.cut(a, bins=boundaries, right=False, labels=False)
a_discretized_2 = pd.cut(a, bins=boundaries, labels=range(len(boundaries) - 1), right=False)
a_discretized_3 = pd.cut(a, bins=boundaries, labels=range(len(boundaries) - 1), right=False).astype(float)
print(a, '\n')
print(a_discretized_1, '\n', a_discretized_1.dtype, '\n')
print(a_discretized_2, '\n', a_discretized_2.dtype, '\n')
print(a_discretized_3, '\n', a_discretized_3.dtype, '\n')

# https://stackoverflow.com/questions/23267767/how-to-do-discretization-of-continuous-attributes-in-sklearn