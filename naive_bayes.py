import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

data = pd.read_csv('src/studi_num.csv')
print("data: ", data)

X = data.iloc[:, :-1].values
y = data.iloc[:, 5].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("y_pred: ", y_pred)
print("np array: ", np.array(y_test))
print("confusion_matrix: ", confusion_matrix(y_test, y_pred))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))