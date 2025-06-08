import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# %matplotlib inline

cancer = load_breast_cancer() # load sklearn dataset
print(cancer.keys())
print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print(df.head())

# standardization dataset
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

# create pca from scaled data
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

print(scaled_data.shape)
print(x_pca.shape)

plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=cancer['target'], cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

plt.show()

print(pca.components_)

# heat maps
df_comp = pd.DataFrame(pca.components_, columns=cancer['feature_names'])
plt.figure(figsize=(12, 6))
sns.heatmap(df_comp,cmap='plasma',)

plt.show()