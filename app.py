import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

# -------------------------------
# 🎨 PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="🚀",
    layout="wide",
)

# -------------------------------
# 💅 CUSTOM CSS (modern gradient + glow)
# -------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    h1 {
        text-align: center;
        color: #00FFFF;
        text-shadow: 0 0 15px #00FFFF;
        font-weight: 800;
        margin-bottom: 10px;
    }
    h2, h3 {
        color: #FFD369;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00ADB5, #007a80);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #00ffff, #00b4d8);
        color: black;
        font-weight: 700;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid rgba(255,255,255,0.2);
    }
    .social-icons {
        margin-top: 0.5rem;
    }
    .icon {
        margin: 0 10px;
        text-decoration: none;
        color: white;
        font-size: 1.2rem;
        transition: 0.3s;
    }
    .icon:hover {
        color: #00FFFF;
        text-shadow: 0 0 10px #00FFFF;
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
st.title("🚀 Customer Churn Prediction Dashboard")
st.markdown("<h3 style='text-align:center;'>Predict churn and visualise customer insights</h3>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# 📂 UPLOAD SECTION
# -------------------------------
uploaded_file = st.file_uploader("📁 Upload your CSV file to begin:", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")
    st.dataframe(data.head(), use_container_width=True)

    # -------------------------------
    # 🧠 MAKE PREDICTIONS (with column alignment)
    # -------------------------------
    try:
        # ✅ Load reference columns from model metadata if saved (or set manually)
        expected_features = list(model.feature_name())

        # ✅ Align incoming data to match training columns only
        missing_cols = [col for col in expected_features if col not in data.columns]
        extra_cols = [col for col in data.columns if col not in expected_features]

        if extra_cols:
            st.warning(f"⚠️ Ignoring {len(extra_cols)} extra columns not seen during training: {extra_cols}")

        for col in missing_cols:
            data[col] = 0  # fill missing with zero safely

        # Keep only expected features in same order
        data = data[expected_features]

        # 🔮 Predict safely
        preds = model.predict(data)

        data["Predicted_Churn"] = preds
        # Scale predictions between 0–1
        churn_rate = np.mean(preds / np.max(preds))


        st.markdown("---")
        st.markdown(f"<h2>📈 Overall Predicted Churn Rate: <span style='color:#00FFFF;'>{churn_rate:.2%}</span></h2>", unsafe_allow_html=True)

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
    st.info("💡 Upload a CSV file to start predictions.")

# -------------------------------
# 🌐 SOCIAL FOOTER WITH LINKS
# -------------------------------
st.markdown("""
<div class="footer">
    <p>Made with ❤️ by <b>Saravanan Vasudevan</b><br>
    MSc Data Science & Analytics | Cardiff University</p>
    <div class="social-icons">
        <a href="https://www.linkedin.com/in/saravanan-vasudevan" target="_blank" class="icon">🔗 LinkedIn</a>
        <a href="https://twitter.com/" target="_blank" class="icon">❌ X</a>
    </div>
</div>
""", unsafe_allow_html=True)
