import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('D:\\dl_data\\ctr\\train_sample.csv')

site_domain = pd.get_dummies(data_train['site_domain'], prefix= 'site_domain')
site_category = pd.get_dummies(data_train['site_category'], prefix= 'site_category')
df = pd.concat([site_domain, site_category], axis=1)
##df.drop(['site_domain', 'site_category'], axis=1, inplace=True)

train_df = df.filter(regex='click|site_domain_.*|site_category_.*|hour|C1|banner_pos')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = linear_model.LogisticRegression(C=1,penalty="l2")
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))