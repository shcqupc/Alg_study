from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
df = pd.read_csv("a8_titanic/data/train.csv")
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
train_df = df.filter(regex='Survived|SibSp|Parch')
#train_df.to_csv("processed_titanic.csv" , encoding = "utf-8")
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]
print(len(X[0]))
clf_3 = RandomForestClassifier()
clf_3.fit(X, y)
 
# Predict on training set
pred_y_3 = clf_3.predict(X)
# How's our accuracy?
print(accuracy_score(y, pred_y_3))
print(confusion_matrix(y, pred_y_3))
labels = clf_3.classes_
conf_df = pd.DataFrame(confusion_matrix(y, pred_y_3), columns=labels, index=labels)
print(conf_df)


# select related data
array_0_0 = []
array_1_0 = []
array_0_1 = []
array_1_1 = []

for i in range(len(y)):
    if y[i] == 0 and pred_y_3[i] == 0:
        array_0_0.append(i)
    if y[i] == 0 and pred_y_3[i] == 1:
        array_0_1.append(i)
    if y[i] == 1 and pred_y_3[i] == 0:
        array_1_0.append(i)
    if y[i] == 1 and pred_y_3[i] == 1:
        array_1_1.append(i)

print(len(y[array_0_0]))
print(len(y[array_0_1]))
print(len(y[array_1_0]))
print(len(y[array_1_1]))

print((X[array_0_0])[0:10])
print((X[array_1_0])[0:10])