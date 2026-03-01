# bank-fraud-detection
#A machine learning web app that detects fraudulent credit card transactions in real-time
---

##  Problem Statement

Banks lose billions of dollars annually to fraudulent transactions. Fraud analysts need an automated system to instantly flag suspicious activity — reducing manual review time and financial loss.

> *"Is this a $10 problem or a $1,000,000 problem?"*
> Catching even one fraud case can save thousands of dollars — massive ROI.

---

##  Pre-MVP Blueprint

| Step | Decision |
|------|----------|
| **Problem** | Fraudulent transactions cost banks billions annually |
| **Who suffers** | Fraud analysts spending hours on manual reviews |
| **Build vs Buy** | Custom ML = more private, accurate, and tailored |
| **Data** | Kaggle Credit Card Fraud Dataset — no PII issues |
| **Tech Stack** | Python + Scikit-learn + Streamlit |
| **MVP Scope** | Input transaction → Predict fraud → Show confidence score |

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| ML Model | Random Forest (Scikit-learn) |
| Imbalance Fix | SMOTE (imbalanced-learn) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit Cloud |


---

## Project Structure

```
bank-fraud-detection/
├── app.py                  ← Streamlit web app
├── fraud_model.ipynb       ← Model training notebook
├── model.pkl               ← Trained Random Forest model
├── scaler.pkl              ← Feature scaler
├── features.pkl            ← Feature names
├── requirements.txt        ← Dependencies
├── sample_10_transactions.csv  ← Sample data for testing
└── README.md
```

---

##  Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest (100 trees) |
| ROC-AUC Score | ~0.99 |
| Dataset Size | 284,807 transactions |
| Fraud Rate | ~0.17% (highly imbalanced) |
| Imbalance Handling | SMOTE oversampling |

---

##  App Features

### 🔢 Manual Input
Enter individual transaction details (Time, Amount, V1–V28) and get an instant fraud prediction with a confidence score and risk meter.

### 📁 Batch CSV Upload
Upload a CSV file with multiple transactions — the app flags all suspicious ones and lets you download the results.

---

##  Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions over 2 days
- **Features:** V1–V28 (PCA-anonymized), Time, Amount
- **Target:** Class (0 = Legitimate, 1 = Fraud)
- **No PII** — all sensitive features are PCA-transformed

---

