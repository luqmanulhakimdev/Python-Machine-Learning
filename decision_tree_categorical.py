import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics

data = pd.read_csv('src/golf_decision_num.txt')
print("data: ", data)

X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

one_hot = pd.get_dummies(data[['Outlook', 'Temp.', 'Humidity', 'Wind']], drop_first=False)
one_hot.head()

data = data.drop(['Outlook', 'Temp.', 'Humidity', 'Wind'], axis = 1)
X = pd.concat([one_hot, data], axis = 1)
print("X head: ", X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
