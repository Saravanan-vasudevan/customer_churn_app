import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction App")

model = joblib.load("customer_churn_model.pkl")

# ✅ Exact 17 feature columns from training
FEATURE_COLUMNS = [
    'store_nbr', 'onpromotion', 'cluster', 'perishable',
    'transactions', 'month', 'day', 'year', 'item_nbr',
    'weekday', 'lag_1', 'lag_7', 'rolling_mean_7',
    'rolling_mean_30', 'promo_days', 'avg_sales_promo', 'price_index'
]

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

    # ✅ Filter columns to match training features
    data = data[[col for col in FEATURE_COLUMNS if col in data.columns]]

    # ✅ Fill missing columns if any
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            data[col] = 0

    # ✅ Reorder columns
    data = data[FEATURE_COLUMNS]

    # ✅ Predict safely
    preds = model.predict(data)
    data["Predicted_Churn"] = preds

    st.subheader("🔮 Predictions (Top 20)")
    st.dataframe(data.head(20))

    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download predictions", csv, "predicted_churn.csv", "text/csv")
else:
    st.info("Upload a CSV file to start prediction.")
