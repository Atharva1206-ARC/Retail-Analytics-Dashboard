import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# 📥 Load data
df = pd.read_csv("streamlit_app/data/customer_features.csv")

# 🎯 Features and target
X = df[["Recency", "Frequency", "DistinctItems"]]
y = df["Monetary"]  # Target: Future purchase amount

# 🔄 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ⚖️ Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🌲 Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 📊 Evaluate
y_pred = model.predict(X_test_scaled)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 💾 Save model and scaler
os.makedirs("streamlit_app/models", exist_ok=True)
joblib.dump(model, "streamlit_app/models/reg_model.pkl")
joblib.dump(scaler, "streamlit_app/models/reg_scaler.pkl")
print("✅ Regression model and scaler saved.")
