import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('src/dataset_enc.csv')
print("data: ", data)

y = OneHotEncoder(sparse_output=False)
y = y.fit_transform(data[['Nama Produk']])
print("array: ", y)

data_y = pd.DataFrame(y)
print("data frame: ", data_y)

header = data['Nama Produk'].sort_values()
data_y.columns = 'Produk_'+header
print("data frame with header: ", data_y)

data_join = data.join(data_y)
print("joined data: ", data_join)
