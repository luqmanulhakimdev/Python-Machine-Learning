import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from chefboost import Chefboost as chef
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression

data = pd.read_csv("src/train.csv")
print("data: ", data)

X = data.iloc[:,0:20]  # independent columns
y = data.iloc[:,-1]    # target column i.e price range

## chi2 method
# apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X, y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  # naming the dataframe columns

print("feature scores: ", featureScores)
print("10 best feature scores: ", featureScores.nlargest(10, 'Score'))  # print 10 best features

## feature importance method
model = ExtraTreesClassifier()
model.fit(X,y)

print("feature importances: ", model.feature_importances_) # use inbuilt class feature_importances of tree based classifiers

# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

## correlation matrix with heatmap
# get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

# plot heat map
g = sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

## ID3 method
data = pd.read_csv('src/golf.txt')
print("data golf: ", data)

config = {'algorithm': 'ID3'}
model = chef.fit(data, config)

def findDecision(obj): #obj[0]: Outlook, obj[1]: Temp., obj[2]: Humidity, obj[3]: Wind
	# {"feature": "Outlook", "instances": 14, "metric_value": 0.9403, "depth": 1}
	if obj[0] == 'Rain':
		# {"feature": "Wind", "instances": 5, "metric_value": 0.971, "depth": 2}
		if obj[3] == 'Weak':
			return 'Yes'
		elif obj[3] == 'Strong':
			return 'No'
		else: return 'No'
	elif obj[0] == 'Sunny':
		# {"feature": "Humidity", "instances": 5, "metric_value": 0.971, "depth": 2}
		if obj[2] == 'High':
			return 'No'
		elif obj[2] == 'Normal':
			return 'Yes'
		else: return 'Yes'
	elif obj[0] == 'Overcast':
		return 'Yes'
	else: return 'Yes'

outlook = 14 * 0.9403 - 5 * 0.971 - 5 * 0.971 
wind = 5 * 0.971
humidity = 5 * 0.971
temperature = 0
total = outlook + wind + humidity + temperature

print ('outlook = ', 100*outlook/total)
print ('wind = ', 100*wind/total)
print ('humidity = ', 100*humidity/total)
print ('temperature = ', 100*temperature/total)

## forward feature selection
data = pd.read_csv('src/golf_label_num.txt')
print("data head: ", data.head())
print("data shape: ", data.shape)
print("data info: ", data.info())
print("data sum null: ", data.isnull().sum())

X = data.drop(['Decision'], axis=1)
y = data['Decision']

print("x shape: ", X.shape)
print("y shape: ", y.shape)

lreg = LinearRegression()
fs = sfs(lreg, k_features=3, forward=True, verbose=2, scoring='neg_mean_squared_error')
fs = fs.fit(X, y)

feat_names = list(fs.k_feature_names_)
print("feat names: ", feat_names)

new_data = data[feat_names]
new_data['Decision'] = data['Decision']

print("data head: ", new_data.head())
