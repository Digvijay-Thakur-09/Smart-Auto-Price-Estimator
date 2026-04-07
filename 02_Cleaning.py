import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

df = pd.read_csv("ford.csv")

# df_coor = df.corr(numeric_only=True)
# plt.figure(figsize=(8,6))
# sns.heatmap(data=df_coor,annot=True,cmap='coolwarm')
# plt.show()

#cleaning data
year = datetime.now().year
df = df[df['year']<year]

df['engineSize'] = df.groupby('model')['engineSize'].transform(lambda x: x.replace(0,x.median()))

df['mpg'] = df.groupby(['model','engineSize'])['mpg'].transform(lambda x: x.mask(x<10,x.median()))

df['mileage'] = df.groupby(['year','engineSize'])['mileage'].transform(lambda x:x.mask(x<100,x.median()))
df['price'] = df.groupby(['model','year','engineSize'])['price'].transform(lambda x:x.mask(x<1000,x.median()))

# print(df[['mpg','price','mileage','engineSize']].head())

#Checking how many rows have data less than the above values;
# bad_rows = df[(df['price'] < 1000) | (df['mileage'] < 100)]
# print(bad_rows)
# print("Number of bad rows:", bad_rows.shape[0])

print(df.shape)
df = df[(df['price']>=1000) & (df['mileage']>=100)]
print(df.shape)

print(df.describe().loc[['min','max']])

print(df.to_csv('ford_clean_data.csv',index=False))