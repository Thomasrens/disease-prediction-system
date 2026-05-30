import streamlit as st
import joblib
import numpy as np
import plotly.express as px

# Page settings
st.set_page_config(
    page_title="MediInsight",
    page_icon="🏥",
    layout="wide"
)

# Load models
diabetes_model = joblib.load("best_model.pkl")
heart_model = joblib.load("heart_model.pkl")

# Sidebar
st.sidebar.title("🏥 MediInsight")
st.sidebar.markdown("AI-Driven Disease Prediction Framework")

selected_disease = st.sidebar.selectbox(
    "Select Disease",
    [
        "Diabetes Prediction",
        "Heart Disease Prediction"
    ]
)
st.title("🏥 MediInsight")
st.subheader("AI-Driven Disease Prediction Framework")
st.markdown("---")
st.sidebar.info(
    """
    AI-powered healthcare prediction system
    using Machine Learning.
    """
)

# =========================
# DIABETES PREDICTION
# =========================

if selected_disease == "Diabetes Prediction":

    st.title("🧠 AI Diabetes Prediction")

    st.write("Enter patient details below")

    st.divider()

    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
    insulin = st.number_input("Insulin Level", min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0, step=1)

    if st.button("Predict Diabetes"):

        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]])

        prediction = diabetes_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Diabetes Detected")
        else:
            st.success("✅ No Diabetes Detected")

# =========================
# HEART DISEASE PREDICTION
# =========================

elif selected_disease == "Heart Disease Prediction":

    st.title("❤️ Heart Disease Prediction")

    st.write("Enter patient details below")

    st.divider()

    age = st.number_input("Age", min_value=0.0)
    sex = st.number_input("Sex (1 = Male, 0 = Female)", min_value=0.0)
    cp = st.number_input("Chest Pain Type", min_value=0.0)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0.0)
    chol = st.number_input("Cholesterol", min_value=0.0)
    fbs = st.number_input("Fasting Blood Sugar", min_value=0.0)
    restecg = st.number_input("Rest ECG", min_value=0.0)
    thalach = st.number_input("Max Heart Rate", min_value=0.0)
    exang = st.number_input("Exercise Induced Angina", min_value=0.0)
    oldpeak = st.number_input("Oldpeak", min_value=0.0)
    slope = st.number_input("Slope", min_value=0.0)
    ca = st.number_input("CA", min_value=0.0)
    thal = st.number_input("Thal", min_value=0.0)

    if st.button("Predict Heart Disease"):

        input_data = np.array([[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]])

        prediction = heart_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease Detected")

# =========================
# ACCURACY CHART
# =========================

st.divider()

st.subheader("📊 Algorithm Accuracy Comparison")

models = [
    "Diabetes SVM",
    "Heart Random Forest"
]

accuracies = [
    76.62,
    98.53
]

fig = px.bar(
    x=models,
    y=accuracies,
    color=models,
    labels={"x": "Models", "y": "Accuracy"},
    title="Model Accuracy Comparison"
)

st.plotly_chart(fig)

st.caption("Developed using Machine Learning and Streamlit")
