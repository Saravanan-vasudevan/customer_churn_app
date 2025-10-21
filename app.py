import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction App")

model = joblib.load("customer_churn_model.pkl")

st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload your CSV file  
2. The model will predict churn/sales for each record  
3. Download the predictions as a CSV
""")

uploaded_file = st.file_uploader("📂 Upload customer data (CSV)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("✅ Data preview:")
    st.dataframe(data.head())

    # 🔧 Convert object columns to category codes (same as training)
    for col in data.select_dtypes("object").columns:
        data[col] = data[col].astype("category").cat.codes

    # 🔧 Force all to numeric
    data = data.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 🔧 Predict safely
    try:
        preds = model.predict(data)
        data["Predicted_Churn"] = preds
        st.subheader("🔮 Predictions (Top 20)")
        st.dataframe(data.head(20))

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download predictions", csv, "predicted_churn.csv", "text/csv")
    except Exception as e:
        st.error("⚠️ Prediction failed — check input format.")
        st.code(str(e))
else:
    st.info("Upload a CSV file to start prediction.")
