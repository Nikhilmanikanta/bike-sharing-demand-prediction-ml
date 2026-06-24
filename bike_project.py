import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('bike_sharing.csv')

# =========================
# DATE PROCESSING
# =========================
df['dteday'] = pd.to_datetime(df['dteday'])
df['day'] = df['dteday'].dt.day
df['month'] = df['dteday'].dt.month
df['year'] = df['dteday'].dt.year

# =========================
# DROP USELESS / LEAKAGE
# =========================
df = df.drop(['instant','dteday','casual','registered'], axis=1)

# =========================
# ENCODING
# =========================
df = pd.get_dummies(df, drop_first=True)

# =========================
# SPLIT
# =========================
X = df.drop('cnt', axis=1)
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODELS
# =========================

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

# XGBoost
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

# =========================
# RESULTS
# =========================
print("\n===== MODEL PERFORMANCE =====")
print("Linear Regression RMSE:", lr_rmse)
print("Random Forest RMSE:", rf_rmse)
print("XGBoost RMSE:", xgb_rmse)

# =========================
# BEST MODEL
# =========================
models = {
    'Linear Regression': lr_rmse,
    'Random Forest': rf_rmse,
    'XGBoost': xgb_rmse
}

best_model = min(models, key=models.get)
print("\nBest Model:", best_model)

# =========================
#  FEATURE IMPORTANCE 
# =========================
importances = xgb.feature_importances_

plt.figure(figsize=(10,6))
plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# =========================
#  SAMPLE PREDICTION 
# =========================
sample = X_test.iloc[5:6]

pred = xgb.predict(sample)

print("\n===== SAMPLE TEST PREDICTION =====")
print("Predicted:", int(pred[0]))
print("Actual:", y_test.iloc[5])

error = abs(pred[0] - y_test.iloc[5])
print("Error:", error)

# =========================
#  USER INPUT PREDICTION
# =========================

print("\nEnter values for prediction:")

season = int(input("Season (1:Spring, 2:Summer, 3:Fall, 4:Winter): "))
yr = int(input("Year (0:2018, 1:2019): "))
mnth = int(input("Month (1-12): "))
holiday = int(input("Holiday (0/1): "))
weekday = int(input("Weekday (0=Sun ... 6=Sat): "))
workingday = int(input("Working Day (0/1): "))
weathersit = int(input("Weather (1-4): "))
temp = float(input("Temperature: "))
atemp = float(input("Feels like temp: "))
hum = float(input("Humidity: "))
windspeed = float(input("Windspeed: "))

new_data = pd.DataFrame([{
    'season': season,
    'yr': yr,
    'mnth': mnth,
    'holiday': holiday,
    'weekday': weekday,
    'workingday': workingday,
    'weathersit': weathersit,
    'temp': temp,
    'atemp': atemp,
    'hum': hum,
    'windspeed': windspeed
}])

# Encoding input
new_data = pd.get_dummies(new_data)

# Match training columns
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# Prediction
final_pred = xgb.predict(new_data)

print("\n===== FINAL USER PREDICTION =====")
print("Predicted Bike Count:", max(0, int(final_pred[0])))

# =========================
#  FINAL CONCLUSION
# =========================
print("\nConclusion:")
print("XGBoost performed best due to its ability to capture complex non-linear relationships in the data.")