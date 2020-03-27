import pandas as pd
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

dfoff = pickle.load(open("../data/dfoff.pickle", "rb"))
print(dfoff['Merchant_id'].value_counts(dropna=False))