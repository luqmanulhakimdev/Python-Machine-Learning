import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

data = pd.read_csv('src/dataset_gaji.csv', sep=',')
data.head(6)

batas_bin = [0, 1400000, 4000000]
kategori = ['Kecil','Besar']
data['gaji_binned_1'] = pd.cut(data['Gaji Orang Tua'], bins=batas_bin, labels=kategori)
print("data: ", data)

# linspace
bins = np.linspace(min(data['Gaji Orang Tua']), max(data['Gaji Orang Tua']), 3)
print("bins: ", bins)
kategori = ['Kecil','Besar']
data['gaji_binned2'] = pd.cut(data['Gaji Orang Tua'], bins=bins, labels=kategori, include_lowest=True)
print("data linspace: ", data)

# quantile
data['Gaji_binned3'] = pd.qcut(data['Gaji Orang Tua'], 2)
print("data quantile: ", data)

kategori = ['Kecil','Besar']
data['Gaji_binned3'] = pd.qcut(data['Gaji Orang Tua'], q=2, labels=kategori)
print("data quantile: ", data)
