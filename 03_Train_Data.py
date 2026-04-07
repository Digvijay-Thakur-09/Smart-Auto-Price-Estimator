import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('ford_clean_data.csv')

X = df.drop(columns=['price'])
y = df['price']

# print(X,y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

cols = X_train.select_dtypes(include='object').columns
encoder = OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore')

X_train_encoded = encoder.fit_transform(X_train[cols])
X_test_encoded = encoder.transform(X_test[cols])

X_train_encoded = pd.DataFrame(X_train_encoded,columns=encoder.get_feature_names_out(cols),index=X_train.index)
X_test_encoded = pd.DataFrame(X_test_encoded,columns=encoder.get_feature_names_out(cols),index=X_test.index)

numeric_cols = ['year','mileage','mpg','engineSize','tax']

X_train_final = pd.concat([X_train[numeric_cols], X_train_encoded], axis=1)
X_test_final = pd.concat([X_test[numeric_cols], X_test_encoded], axis=1)

num_cols =  X_train.select_dtypes(include='number').columns

scaler = StandardScaler()

X_train_final[num_cols] = scaler.fit_transform(X_train_final[num_cols])
X_test_final[num_cols] = scaler.transform(X_test_final[num_cols])

# Linear
linear_model = LinearRegression()
linear_model.fit(X_train_final, y_train)

# Decision Tree
decision_model = DecisionTreeRegressor()
decision_model.fit(X_train_final, y_train)

# KNN
knn_model = KNeighborsRegressor(n_neighbors=7)
knn_model.fit(X_train_final, y_train)

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train_final, y_train)


import pickle

# Save all models
models = {
    "Linear": linear_model,
    "Decision Tree": decision_model,
    "KNN": knn_model,
    "Random Forest": rf_model
}

# =========================
# PREDICTIONS
# =========================

y_pred_linear = linear_model.predict(X_test_final)
y_pred_decision = decision_model.predict(X_test_final)
y_pred_knn = knn_model.predict(X_test_final)
y_pred_rf = rf_model.predict(X_test_final)

# =========================
# METRICS
# =========================

# Linear
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Decision Tree
mae_decision = mean_absolute_error(y_test, y_pred_decision)
mse_decision = mean_squared_error(y_test, y_pred_decision)
r2_decision = r2_score(y_test, y_pred_decision)

# KNN
mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

# Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

metrics = {
    "Linear": {"MAE": mae_linear, "MSE": mse_linear, "R2": r2_linear},
    "Decision Tree": {"MAE": mae_decision, "MSE": mse_decision, "R2": r2_decision},
    "KNN": {"MAE": mae_knn, "MSE": mse_knn, "R2": r2_knn},
    "Random Forest": {"MAE": mae_rf, "MSE": mse_rf, "R2": r2_rf}
}

with open("metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

with open("metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

with open("models.pkl", "wb") as f:
    pickle.dump(models, f)

# Save preprocessing
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature column order
with open("columns.pkl", "wb") as f:
    pickle.dump(X_train_final.columns.tolist(), f)

# Save categorical columns
with open("cat_cols.pkl", "wb") as f:
    pickle.dump(cols, f)

with open("num_cols.pkl", "wb") as f:
    pickle.dump(num_cols.tolist(), f)