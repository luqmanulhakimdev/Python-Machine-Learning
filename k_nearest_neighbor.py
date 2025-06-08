import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('src/gizi.csv')
print("data: ", data)

X = data.iloc[:, :-1].values
y = data.iloc[:, 5].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
