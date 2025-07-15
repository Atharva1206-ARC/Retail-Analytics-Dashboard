import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

# 📥 Load data
df = pd.read_csv("streamlit_app/data/customer_features.csv")

# ✅ Ensure target column exists
if "HighValue" not in df.columns:
    df["HighValue"] = (df["Monetary"] > df["Monetary"].median()).astype(int)

# 🎯 Features and target
X = df[["Recency", "Frequency", "Monetary", "DistinctItems"]]
y = df["HighValue"]

# 🔄 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔃 Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🌲 Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 📊 Print evaluation
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# 💾 Save model and scaler
os.makedirs("streamlit_app/models", exist_ok=True)
joblib.dump(model, "streamlit_app/models/rf_model.pkl")
joblib.dump(scaler, "streamlit_app/models/scaler.pkl")
print("✅ Model and scaler saved successfully.")
