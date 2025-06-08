import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
sns.set(color_codes=True)

df = pd.read_csv("src/data_car.csv")
print(df.head(5)) # to display the top 5 row
print(df.tail(5)) # to display the last 5 row
print(df.dtypes) # to display the type column

# rename header
df = df.rename(columns={"Engine HP": "HP", "Engine Cylinders": "Cylinders", "Transmission Type": "Transmission", "Driven_Wheels": "Drive Mode","highway MPG": "MPG-H", "city mpg": "MPG-C", "MSRP": "Price" })
print(df.head(5))

print(df.shape) # display total row & column (total_row, total_column)

## delete duplicate row
duplicate_rows = df[df.duplicated()]
print("total duplicate row: ", duplicate_rows.shape)

print("count total row: ", df.count())
df = df.drop_duplicates() # remove duplicates row
print("count total row: ", df.count())

## delete row when data isnull
print("total null data: ", df.isnull().sum())

df = df.dropna()
print("count total row: ", df.count())
print("total null data: ", df.isnull().sum())


## detecting outliers
f = plt.figure(figsize=(12, 4))
f.add_subplot(1, 3, 1)
df['HP'].plot(kind='kde')
f.add_subplot(1, 3, 2)
plt.boxplot(df['HP'])

f.add_subplot(1, 3, 3)
sns.boxplot(x=df['Price'])
# sns.boxplot(x=df['HP'])
# sns.boxplot(x=df['Cylinders'])
plt.show() # need to explicitly call for display plot

print("describe price: ", df.Price.describe())

# histogram
df.Make.value_counts().nlargest(40).plot(kind='bar', figsize=(10, 5))
plt.title("Number of cars by make")
plt.ylabel("Number of cars")
plt.xlabel("Make")

plt.show()

# heat maps
plt.figure(figsize=(10, 5))
c = df.corr(numeric_only=True)
sns.heatmap(c, cmap="RdYlGn", annot=True)
print("heat maps: ", c)
plt.show()

result = pd.pivot_table(data=df, index='Transmission', columns='Drive Mode',values='Price')
print("heat maps: ", result)

sns.heatmap(result, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()

# scatterplot
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['HP'], df['Price'])
ax.set_xlabel('HP')
ax.set_ylabel('Price')
plt.show()

sns.pairplot(data = df, vars=['HP','Drive Mode','Price'])
plt.show()

# correlation matrix
print("correlation matrix: ", df[['HP','Price']].corr())
