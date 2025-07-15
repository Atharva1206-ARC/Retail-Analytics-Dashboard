import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("streamlit_app/data/customer_features.csv")
X = df[["Recency", "Frequency", "DistinctItems"]]
y = df["Monetary"]

# Load model & scaler
model = joblib.load("streamlit_app/models/reg_model.pkl")
scaler = joblib.load("streamlit_app/models/reg_scaler.pkl")

# Predict
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

# Metrics
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"📉 Mean Squared Error (MSE): {mse:,.2f}")
print(f"📉 Mean Absolute Error (MAE): {mae:,.2f}")
print(f"📈 R² Score: {r2:.4f}")

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.4, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual Spend")
plt.ylabel("Predicted Spend")
plt.title("Actual vs Predicted Customer Spend")
plt.grid(True)
plt.tight_layout()
plt.show()
