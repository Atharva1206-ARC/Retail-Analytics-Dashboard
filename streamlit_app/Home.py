import streamlit as st

# 🔧 Page configuration
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🛒",
    layout="wide"
)

# 🏷️ Title
st.title("🛍️ Retail Analytics Dashboard")

# 🌟 Intro Message
st.markdown("""
Welcome to the **Retail Analytics Dashboard** — your one-stop solution to explore, predict, and optimize customer behavior in the retail space.

---

### 🔎 What You Can Do Here:
- **Explore** customer purchase behavior through EDA & segmentation.
- **Predict** high-value customers using machine learning.
- **Analyze** future spend and product preferences.
- **Recommend** products based on past purchases.
- **Evaluate** your models with intuitive visual reports.

---

""")

# 🎯 Instructions
st.info("👉 Use the sidebar on the left to navigate through different modules.")

# 🖼️ Optional Image or Banner
# st.image("streamlit_app/assets/banner.png", use_column_width=True)

# 📌 Footer
st.markdown("""
---
Made by :\n 
Siddhant Tilak - 20220802421\n
Atharva Chavan - 20220802348 \n
Harsh Kakad - 20220802342 
""")
