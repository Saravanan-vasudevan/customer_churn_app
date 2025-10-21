import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

# -------------------------------
# 🎨 PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📊",
    layout="wide",
)

# -------------------------------
# ✨ CUSTOM CSS (modern glass look)
# -------------------------------
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #fff;
    }
    .main {
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(6.3px);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    h1, h2, h3 {
        color: #00ADB5;
        text-align: center;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #00ADB5;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #007a80;
        transform: scale(1.05);
    }
    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 📦 LOAD MODEL
# -------------------------------
model = joblib.load("customer_churn_model.pkl")

# -------------------------------
# 🏷️ APP TITLE
# -------------------------------
st.title("📊 Customer Churn Prediction App")
st.subheader("Upload customer data and predict churn probability")

# -------------------------------
# 📂 UPLOAD SECTION
# -------------------------------
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.markdown("✅ **Data uploaded successfully!**")
    
    st.dataframe(data.head(), use_container_width=True)

    # -------------------------------
    # 🧠 MAKE PREDICTIONS
    # -------------------------------
    try:
        preds = model.predict(data)
        data["Predicted_Churn"] = preds

        # 🧾 METRICS SUMMARY
        churn_rate = np.mean(preds)
        st.markdown("---")
        st.markdown(f"### ⚙️ Overall Predicted Churn Rate: **{churn_rate:.2%}**")

        st.download_button(
            label="📥 Download Predictions as CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="predicted_churn.csv",
            mime="text/csv",
        )

        st.success("✅ Predictions generated successfully!")

    except Exception as e:
        st.error("⚠️ Prediction failed — check input format or missing columns.")
        st.exception(e)

else:
    st.info("📤 Upload a CSV file to begin.")

# -------------------------------
# 🧾 FOOTER
# -------------------------------
st.markdown("""
<hr>
<div style='text-align:center;'>
    Made with ❤️ by <b>Saravanan Vasudevan</b>  
    <br>
    MSc Data Science & Analytics | Cardiff University
</div>
""", unsafe_allow_html=True)
