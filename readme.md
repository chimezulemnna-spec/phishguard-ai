# 🛡️ PhishGuard AI

A hybrid machine learning-based phishing website detection system that analyzes URLs and webpage content in real time to identify phishing threats.

Built as a final year project at **Benson Idahosa University**, Department of Computer Science, 2026.

---

## 🔍 Overview

PhishGuard AI combines two ensemble models to detect phishing websites with over **99.85% accuracy**. It extracts 38 features from the URL structure and live webpage content, then runs them through trained machine learning models to produce an instant verdict.

---

## ✨ Features

- **Dual ensemble detection** — Original (RF + XGB) and PILWD (RF + XGB + GB) models run in parallel
- **38-feature extraction** — 25 URL features + 13 live content features
- **SHAP explainability** — Visual breakdown of why the model made its decision
- **Google Safe Browsing API** — Cross-checks every URL against Google's threat database
- **WHOIS + SSL checks** — Domain age, registrar and certificate validity
- **Global Phishing Threat Map** — Interactive D3.js world map showing top phishing origin countries
- **Model Evaluation page** — Confusion matrix and ROC curve on live dataset samples
- **Model comparison** — Side-by-side confidence scores from both ensembles
- **Scan history** — Last 5 scans with verdict and confidence
- **Export reports** — Download results as TXT or JSON
- **Mobile responsive** — Full hamburger nav and adaptive layout

---

## 🤖 Models & Performance

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Random Forest | 99.87% | 99.98% | 99.75% | 99.94% |
| XGBoost | 99.86% | 99.98% | 99.74% | 99.95% |
| Original Voting Ensemble (RF + XGB) | 99.86% | 99.98% | 99.75% | 99.95% |
| PILWD Weighted Ensemble (RF + XGB + GB) | 99.65% | — | — | 99.65% |

Trained on **128,408 URLs** (50/50 balanced — legitimate and phishing).

---

## 🧱 Tech Stack

- **Frontend/App** — Streamlit
- **ML Models** — scikit-learn, XGBoost
- **Explainability** — SHAP
- **Visualizations** — D3.js, Matplotlib, TopoJSON
- **Data** — pandas, NumPy
- **APIs** — Google Safe Browsing API, WHOIS, SSL

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/chimezulemnna-spec/phishguard-ai.git
cd phishguard-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
phishguard-ai/
├── app.py                  # Main Streamlit application
├── styles.css              # Custom dark theme styles
├── requirements.txt
├── models/
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── voting_model.pkl
│   ├── scaler.pkl
│   ├── weighted_ensemble_pilwd.pkl
│   ├── scaler_pilwd.pkl
│   └── feature_mapping_pilwd.pkl
├── src/
│   ├── url_features.py     # URL feature extraction (25 features)
│   └── content_features.py # Webpage content extraction (13 features)
└── dataset/
    └── combined_urls.csv   # 128k training dataset (not included in repo)
```

---

## 📊 Dataset

The model was trained on a combined dataset of **128,408 URLs** assembled from multiple public phishing and legitimate URL datasets, balanced 50/50 between classes.

The dataset is not included in this repository due to size. The model evaluation page samples from it if available locally.

---

## 👨‍💻 Author

**Ernest Chimezulem Nna**  
Department of Computer Science  
Benson Idahosa University  
2026

---

## 📄 License

This project was developed for academic purposes. All rights reserved © 2026 Ernest Chimezulem Nna.