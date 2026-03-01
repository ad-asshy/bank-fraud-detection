import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Fraud Detector",
    page_icon="",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #ffffff; }
    .fraud-box {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .legit-box {
        background: linear-gradient(135deg, #00c85222, #00c85244);
        border: 2px solid #00c852;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .section-header {
        font-size: 18px;
        font-weight: bold;
        color: #a0aec0;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("##  Bank Fraud Detection System")
st.markdown("*Powered by Machine Learning — Random Forest Classifier*")
st.divider()

# ── Model Stats ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><h3>~99%</h3><p>ROC-AUC Score</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3>284K+</h3><p>Transactions Trained On</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h3>SMOTE</h3><p>Imbalance Handling</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><h3>Real-time</h3><p>Fraud Prediction</p></div>', unsafe_allow_html=True)

st.divider()

# ── Input Mode ────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([" Manual Input", " Upload CSV"])

with tab1:
    st.markdown("### Enter Transaction Details")
    st.info(" V1–V28 are PCA-anonymized features from the bank. Enter the values from your transaction record.")

    col_l, col_r = st.columns(2)

    with col_l:
        time_val = st.number_input(" Time (seconds since first transaction)", value=0.0)
        amount_val = st.number_input(" Amount ($)", value=100.0, min_value=0.0)
        st.markdown("**V1 – V14**")
        v_vals = []
        cols = st.columns(2)
        for i in range(1, 15):
            v_vals.append(cols[(i-1) % 2].number_input(f"V{i}", value=0.0, key=f"v{i}"))

    with col_r:
        st.markdown("**V15 – V28**")
        cols2 = st.columns(2)
        for i in range(15, 29):
            v_vals.append(cols2[(i-15) % 2].number_input(f"V{i}", value=0.0, key=f"v{i}"))

    if st.button(" Analyze Transaction", use_container_width=True, type="primary"):
        amount_scaled = scaler.transform(np.array([[amount_val]]))[0][0]
        time_scaled = (time_val - 94813.86) / 47488.15  # approx normalization

        features = np.array(v_vals + [amount_scaled, time_scaled]).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        st.divider()
        st.markdown("###  Prediction Result")

        fraud_prob = probability[1] * 100
        legit_prob = probability[0] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="fraud-box">
                <h1> FRAUD DETECTED</h1>
                <h2>Fraud Probability: {fraud_prob:.2f}%</h2>
                <p>This transaction has been flagged as potentially fraudulent. Please review immediately.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="legit-box">
                <h1> LEGITIMATE</h1>
                <h2>Legitimate Probability: {legit_prob:.2f}%</h2>
                <p>This transaction appears to be legitimate. No action required.</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric(" Legitimate Score", f"{legit_prob:.2f}%")
        c2.metric(" Fraud Score", f"{fraud_prob:.2f}%")

        # Risk meter
        st.markdown("**Risk Level:**")
        st.progress(fraud_prob / 100)
        if fraud_prob < 30:
            st.success(" Low Risk")
        elif fraud_prob < 70:
            st.warning(" Medium Risk — Review Recommended")
        else:
            st.error(" High Risk — Immediate Action Required")

with tab2:
    st.markdown("### Upload a CSV file for batch prediction")
    st.info("CSV must have columns: Time, V1–V28, Amount")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df):,} transactions")

        if st.button(" Predict All", type="primary"):
            df['Amount_scaled'] = scaler.transform(df[['Amount']].values.reshape(-1, 1))
            df['Time_scaled'] = (df['Time'] - 94813.86) / 47488.15

            feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']
            X = df[feature_cols]

            preds = model.predict(X)
            probas = model.predict_proba(X)[:, 1]

            df['Prediction'] = ['FRAUD' if p == 1 else 'Legitimate' for p in preds]
            df['Fraud_Probability_%'] = (probas * 100).round(2)

            fraud_count = sum(preds)
            st.error(f" {fraud_count} Fraudulent transactions found out of {len(df):,}")
            st.success(f" {len(df) - fraud_count:,} Legitimate transactions")

            result_df = df[['Time', 'Amount', 'Prediction', 'Fraud_Probability_%']]
            st.dataframe(result_df, use_container_width=True)

            csv = result_df.to_csv(index=False)
            st.download_button(" Download Results", csv, "fraud_predictions.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()

