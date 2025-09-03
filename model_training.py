# save_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("house_prices.csv")

# Features & target
features = ['LotArea', 'OverallQual', 'YearBuilt', 'GrLivArea', 'FullBath', 'GarageCars', 'GarageArea']
X = df[features]
y = df['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest (better performance than Linear Regression)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "house_price_model.pkl")
print("✅ Model saved as house_price_model.pkl")
