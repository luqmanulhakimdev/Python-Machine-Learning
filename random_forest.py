import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('src/covid19.csv')
print("data: ", data)

X = data.iloc[:, :-1].values
y = data.iloc[:, 3].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = RandomForestClassifier(criterion="gini", n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(model.feature_importances_) 
