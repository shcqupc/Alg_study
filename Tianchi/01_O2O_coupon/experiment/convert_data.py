import pickle

dfoff = pickle.load(open('../data/dfoff_2.pickle', 'rb'))
dftest = pickle.load(open('../data/dftest_2.pickle', 'rb'))

dfoff.loc[:, 'discount_rate'] = dfoff['discount_rate'].astype(float)
dftest.loc[:, 'discount_rate'] = dftest['discount_rate'].astype(float)
print(dfoff['discount_rate'].unique())
print(dftest['discount_rate'].unique())
dfoff.to_pickle('../data/dfoff_2.pickle')
dftest.to_pickle('../data/dftest_2.pickle')