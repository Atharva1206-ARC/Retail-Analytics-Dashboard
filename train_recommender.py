import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import os

# 📥 Load retail data
df = pd.read_excel("streamlit_app/data/Online Retail.xlsx")
df.dropna(subset=["InvoiceNo", "Description", "CustomerID"], inplace=True)

# 🔄 Clean
df["Description"] = df["Description"].str.strip().str.lower()
df["InvoiceNo"] = df["InvoiceNo"].astype(str)

# 🧺 Create user-product matrix
basket = df.groupby(["CustomerID", "Description"])["Quantity"].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# 🤝 Compute item similarity
item_similarity = pd.DataFrame(cosine_similarity(basket.T),
                               index=basket.columns,
                               columns=basket.columns)

# 💾 Save similarity matrix
os.makedirs("streamlit_app/models", exist_ok=True)
joblib.dump(item_similarity, "streamlit_app/models/item_similarity.pkl")
print("✅ Recommender matrix saved.")
