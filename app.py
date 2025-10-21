import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction App")

# Load model
model = joblib.load("customer_churn_model.pkl")

st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload your CSV file  
2. The model will predict churn probability for each record  
3. Download the predictions as a CSV
""")

uploaded_file = st.file_uploader("📂 Upload customer data (CSV)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("✅ Data preview:")
    st.dataframe(data.head())

    preds = model.predict(data)
    data["Predicted_Churn"] = preds

    st.subheader("🔮 Predictions (Top 20)")
    st.dataframe(data.head(20))

    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download predictions", csv, "predicted_churn.csv", "text/csv")

    # Optional: show saved metrics
    try:
        metrics = pd.read_csv("customer_churn_metrics.csv")
        st.markdown("### 📈 Model Performance")
        st.dataframe(metrics)
    except Exception:
        st.info("Metrics file not found.")
else:
    st.info("Upload a CSV file to start prediction.")
