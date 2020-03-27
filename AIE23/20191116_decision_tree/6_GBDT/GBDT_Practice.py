import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from sklearn.preprocessing import StandardScaler
plt.figure(figsize=[5, 3.5])
ds = pd.read_csv("../0_DT_Tree/data/DS_Adaboost.csv")
ds.Decision = np.where(ds.Decision == 1, 1, -1)


plt.show()
