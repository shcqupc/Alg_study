import numpy as np
import sklearn.preprocessing as preprocessing

X = np.array([[-3., 5., 15],
              [-1., 6., 14],
              [6., 3., 11]])
est_onehot = preprocessing.KBinsDiscretizer(n_bins=[3, 3,3], encode='onehot-dense').fit(X)
est_ordinal = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal').fit(X)
print("est_onehot", est_onehot.transform(X), sep='\n')
print("est_ordinal", est_ordinal.transform(X), sep='\n')
