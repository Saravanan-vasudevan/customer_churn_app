# ☁️ Cloud-Deployed Customer Churn Prediction App

An end-to-end **machine learning microservice** for customer churn prediction, built using **LightGBM** and deployed on **Streamlit Cloud**.  
Containerised with **Docker** and integrated with **GitHub Actions** for automated CI/CD.

---

## 🚀 Features
- Trained on 6 M + retail records using LightGBM
- RMSE ≈ 21.8  |  R² ≈ 0.35
- Real-time churn prediction via Streamlit UI
- Containerised with Docker for portable deployment
- Automated build & test pipeline with GitHub Actions

---

## 🧱 Tech Stack
`Python`  `LightGBM`  `Scikit-learn`  `Streamlit`  `Docker`  `GitHub Actions`

---

## ☁️ Architecture
![Cloud Architecture](architecture.png)

---

## 🔗 Live Demo
👉 [Open on Streamlit Cloud](https://YOUR-APP-LINK.streamlit.app)

---

## 🧰 Run Locally (Docker)
```bash
docker build -t churn-app .
docker run -p 8501:8501 churn-app
