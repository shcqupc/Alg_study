# coding: utf-8
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

date_parser = lambda x: pd.datetime.strptime(x, '%y%m%d%H')

data_types = {
    'id': np.str,
    'click': np.bool_,
    'hour': np.str,
    'C1': np.uint16,
    'banner_pos': np.uint16,
    'site_id': np.object,
    'site_domain': np.object,
    'site_category': np.object,
    'app_id': np.object,
    'app_domain': np.object,
    'app_category': np.object,
    'device_id': np.object,
    'device_ip': np.object,
    'device_model': np.object,
    'device_type': np.uint16,
    'device_conn_type': np.uint16,
    'C14': np.uint16,
    'C15': np.uint16,
    'C16': np.uint16,
    'C17': np.uint16,
    'C18': np.uint16,
    'C19': np.uint16,
    'C20': np.uint16,
    'C21': np.uint16    
}

train_df = pd.read_csv('E:\\dl_data\\ctr\\train_sample.csv',
                           dtype=data_types,
                           parse_dates=['hour'],
                           date_parser=date_parser)

#print(classification_report(y_test, predictions))
test_df = pd.read_csv('E:\\dl_data\\ctr\\test_sample.csv',
                          dtype=data_types,
                          parse_dates=['hour'],
                          date_parser=date_parser)

train_df.info()

X_train, X_test, y_train, y_test = train_test_split(
    train_df.filter(regex='banner_pos|device_type').values, 
    train_df.filter(regex='click').values,
    test_size=0.3,
    random_state=42
)

clf = linear_model.LogisticRegression(C=100.0, penalty='l1', tol=1e-6)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))

