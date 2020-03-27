import numpy as np
def pcc(X, Y):
   ''' Compute Pearson Correlation Coefficient. '''
   # Normalise X and Y
   X -= X.mean()
   Y -= Y.mean()
   # Standardise X and Y
   X /= X.std()
   Y /= Y.std()
   # Compute mean product
   return np.mean(X*Y)

# Using it on a random example
from random import random
X = np.array([random() for x in range(100)])
Y = np.array([random() for x in range(100)])
print(pcc(X, Y))