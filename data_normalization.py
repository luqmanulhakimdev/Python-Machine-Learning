import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv ('src/dataset_gaji.csv')
print("data: ", data)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[['Gaji Orang Tua', 'Umur']])
print("scaled data: ", scaled)
