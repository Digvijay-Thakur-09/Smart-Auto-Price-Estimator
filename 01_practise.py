import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score

df = pd.read_csv("ford.csv")

#First Look towards data
# print(df.head())
# print(df.tail())
# print(df.info())
# print(df.describe())
# print(df.shape)

#histogram
# for col in df.columns:
#     plt.figure(figsize=(8,6))
#     sns.histplot(df[col],kde=True)
#     plt.title(f"Distribution of {col}")
#     plt.show()

#correlation and heatmap
# plt.figure(figsize=(8,6))
# sns.heatmap(df.corr(numeric_only=True),annot=True)
# plt.show()

#boxplot
# plt.figure(figsize=(8,6))
# sns.boxplot(data=df,x='year',y='price')
# plt.xticks(rotation=90)
# plt.show()

#scatterplot
# plt.figure(figsize=(8,6))
# sns.scatterplot(data=df,x='mileage',y='price')
# plt.show()

#box plot
# plt.figure(figsize=(8,6))
# sns.boxplot(data=df,x='engineSize',y='price')
# plt.show()
# #box plot
# plt.figure(figsize=(8,6))
# sns.boxplot(data=df,x='transmission',y='price')
# plt.show()

# #boxplot
# plt.figure(figsize=(8,6))
# sns.boxplot(data=df,x='fuelType',y='price')
# plt.show() 

# #boxplot
# plt.figure(figsize=(8,6))
# sns.boxplot(data=df,x='model',y='price')
# plt.xticks(rotation=90)
# plt.show()

# #boxplot
# plt.figure(figsize=(8,6))
# sns.boxplot(data=df,x='tax',y='price')
# plt.xticks(rotation=90)
# plt.show()
  
#Scatterplot
# plt.figure(figsize=(8,6))
# sns.scatterplot(data=df,x='mpg',y='price')
# plt.xticks(rotation=90)
# plt.show() 

X = df.drop(columns=['price'])
y = df['price']

# print(X)
# print(y)

X_one_encode = pd.get_dummies(X,columns=['model','transmission','fuelType'],drop_first=True)

# print(Xlabel)

#To get numerical columns
num_cols = X_one_encode.select_dtypes(include=['int64', 'float64']).columns
# print(num_cols)


scaler = StandardScaler()
X_one_encode[num_cols] = scaler.fit_transform(X_one_encode[num_cols])

# print(X_one_encode)

X_train,X_test,y_train,y_test = train_test_split(X_one_encode,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_predict = model.predict(X_test)

r2 =  r2_score(y_test,y_predict)

print(f"The r2 score: {r2}")

n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - ((1-r2)*(n-1))/(n-p-1)
print(f"Adjusted R2: {adjusted_r2}")
