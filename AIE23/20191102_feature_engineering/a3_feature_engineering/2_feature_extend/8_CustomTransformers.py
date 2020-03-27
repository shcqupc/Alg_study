import numpy as np
from sklearn.preprocessing import FunctionTransformer
ftr= FunctionTransformer(np.log1p, validate=True)
# ftr= FunctionTransformer(np.log1p, validate=True)
X = np.array([[0, 1], [2, 3]])
print(ftr.transform(X))