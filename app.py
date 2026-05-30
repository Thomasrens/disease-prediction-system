import plotly.express as px
import streamlit as st
import joblib
import numpy as np
import os

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="MediInsight",
    page_icon="🏥",
    layout="wide"
)

# 2. Custom CSS
st.markdown("""
<style>
/* Main Title */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #4CAF50;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 22px;
    color: #B0BEC5;
}

/* Welcome Card */
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #333333;
    margin-bottom: 20px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# 3. Robust Model Loading with Error Handling & Caching
@st.cache_resource
def load_models():
    d_model, h_model = None, None
    try:
        if os.path.exists("best_model.pkl"):
            d_model = joblib.load("best_model.pkl")
        if os.path.exists("heart_model.pkl"):
            h_model = joblib.load("heart_model.pkl")
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return d_model, h_model

diabetes_model, heart_model = load_models()

# 4. Sidebar Configuration
st.sidebar.title("🏥 MediInsight")
st.sidebar.markdown("AI-Driven Disease Prediction Framework")

selected_disease = st.sidebar.selectbox(
    "Select Disease Module",
    [
        "🩸 Diabetes Risk Assessment",
        "❤️ Heart Disease Risk Assessment"
    ]
)

# 5. Header Section
st.markdown("""
<div class="main-title">
🏥 MediInsight
</div>
<div class="sub-title">
AI-Driven Disease Prediction Framework
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div class="card">
<h3>Welcome to MediInsight</h3>
<p>
This AI-powered healthcare platform predicts Diabetes and Heart Disease risk
using Machine Learning algorithms. Please fill out the patient metrics below.
</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DIABETES MODULE
# ---------------------------------------------------------
if selected_disease == "🩸 Diabetes Risk Assessment":
    st.header("🩸 Diabetes Risk Assessment")
    
    if diabetes_model is None:
        st.warning("⚠️ Diabetes model ('best_model.pkl') not found. Please ensure the file is in the directory.")
    else:
        # Organize inputs into columns for better UX
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=0)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, value=20.0)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
            
        with col2:
            glucose = st.number_input("Glucose Level", min_value=0.0, value=120.0)
            insulin = st.number_input("Insulin Level", min_value=0.0, value=79.0)
            age = st.number_input("Age", min_value=1, step=1, value=30)
            
        with col3:
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, value=70.0)
            bmi = st.number_input("BMI", min_value=0.0, value=25.0)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Predict Diabetes Risk"):
            input_data = np.array([[
                pregnancies, glucose, blood_pressure, skin_thickness, 
                insulin, bmi, dpf, age
            ]])

            prediction = diabetes_model.predict(input_data)

            if prediction[0] == 1:
                st.error("⚠️ High Risk of Diabetes Detected. Please consult a physician.")
            else:
                st.success("✅ Low Risk of Diabetes Detected.")

# ---------------------------------------------------------
# HEART DISEASE MODULE
# ---------------------------------------------------------
elif selected_disease == "❤️ Heart Disease Risk Assessment":
    st.header("❤️ Heart Disease Risk Assessment")
    
    if heart_model is None:
        st.warning("⚠️ Heart Disease model ('heart_model.pkl') not found. Please ensure the file is in the directory.")
    else:
        # Organize inputs into columns for better UX
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=1, step=1, value=50)
            sex_val = st.selectbox("Sex", ["Male", "Female"])
            sex = 1 if sex_val == "Male" else 0
            cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
            trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=0.0, value=120.0)
            chol = st.number_input("Cholesterol (mg/dl)", min_value=0.0, value=200.0)
            
        with col2:
            fbs_val = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            fbs = 1 if fbs_val == "Yes" else 0
            restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
            thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0.0, value=150.0)
            exang_val = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            exang = 1 if exang_val == "Yes" else 0
            
        with col3:
            oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, value=1.0)
            slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
            ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
            thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Predict Heart Disease Risk"):
            input_data = np.array([[
                age, sex, cp, trestbps, chol, fbs, restecg, 
                thalach, exang, oldpeak, slope, ca, thal
            ]])

            prediction = heart_model.predict(input_data)

            if prediction[0] == 1:
                st.error("⚠️ High Risk of Heart Disease Detected. Please consult a cardiologist.")
            else:
                st.success("✅ Low Risk of Heart Disease Detected.")

# ---------------------------------------------------------
# FOOTER & ACCURACY CHART
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Model Accuracy Comparison")

models = ["Diabetes (SVM)", "Heart Disease (Random Forest)"]
accuracies = [76.62, 98.53]

fig = px.bar(
    x=models,
    y=accuracies,
    labels={"x": "Models", "y": "Accuracy (%)"},
    title="Machine Learning Model Accuracy Comparison",
    color=accuracies,
    color_continuous_scale="Viridis"
)
fig.update_layout(yaxis_range=[0, 100]) # Locks the Y-axis to 100% for better perspective

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("MediInsight | AI-Driven Disease Prediction Framework")
