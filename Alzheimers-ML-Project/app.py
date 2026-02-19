import streamlit as st
import numpy as np
import joblib
import os

MODEL_PATH = "Alzheimers-ML-Project/models/best_model.pkl"
SCALER_PATH = "Alzheimers-ML-Project/models/scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error("❌ Model files could not be loaded.")
    st.write(e)
    st.stop()


model = joblib.load("Alzheimers-ML-Project/models/best_model.pkl")
scaler = joblib.load("Alzheimers-ML-Project/models/scaler.pkl")


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🧠 Alzheimer’s Disease Risk Prediction Dashboard</h1>
    <p style="text-align:center; font-size:18px;">
    A Machine Learning Tool for Early Risk Detection
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.title("📌 Patient Inputs")

age = st.sidebar.slider("Age", 60, 90, 75)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender = 0 if gender == "Male" else 1

bmi = st.sidebar.slider("BMI", 15.0, 40.0, 25.0)

diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
diabetes = 0 if diabetes == "No" else 1

depression = st.sidebar.selectbox("Depression", ["No", "Yes"])
depression = 0 if depression == "No" else 1

physical = st.sidebar.slider("Physical Activity (hrs/week)", 0, 10, 3)

sleep = st.sidebar.slider("Sleep Quality (4–10)", 4, 10, 7)

mmse = st.sidebar.slider("MMSE Score (0–30)", 0, 30, 20)

memory = st.sidebar.selectbox("Memory Complaints", ["No", "Yes"])
memory = 0 if memory == "No" else 1

# -----------------------------
# Main Layout
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Predict Alzheimer’s Risk")

    if st.button("Run Prediction"):

        sample = np.array([[age, gender, bmi,
                            diabetes, depression,
                            physical, sleep,
                            mmse, memory]])

        sample_scaled = scaler.transform(sample)

        pred = model.predict(sample_scaled)[0]
        prob = model.predict_proba(sample_scaled)[0][1]

        st.subheader("📊 Prediction Result")

        # Risk Meter
        st.progress(float(prob))

        if pred == 1:
            st.error(f"⚠ High Risk Detected\n\nProbability: {prob:.2f}")
        else:
            st.success(f"✅ Low Risk Detected\n\nProbability: {prob:.2f}")

        st.info("This is an AI-based prediction, not a confirmed medical diagnosis.")

with col2:
    st.subheader("📌 Patient Summary")

    st.write(f"**Age:** {age}")
    st.write(f"**BMI:** {bmi}")
    st.write(f"**MMSE Score:** {mmse}")
    st.write(f"**Physical Activity:** {physical} hrs/week")
    st.write(f"**Sleep Quality:** {sleep}/10")

    st.divider()

    st.subheader("🧠 Key Factors")
    st.caption("""
    - Cognitive decline (MMSE)
    - Lifestyle factors (activity, sleep)
    - Medical conditions (diabetes, depression)
    """)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:14px;">
    Developed as an ML Healthcare Project | Alzheimer’s Risk Classification
    </p>
    """,
    unsafe_allow_html=True
)


