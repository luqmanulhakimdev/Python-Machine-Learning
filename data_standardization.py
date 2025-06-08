import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv ('src/dataset_gaji.csv')

scaler = StandardScaler()
scaled = scaler.fit_transform(data[['Gaji Orang Tua', 'Umur']])
print("scaled data: ", scaled)