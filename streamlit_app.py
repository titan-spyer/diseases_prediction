import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Multi-Disease Prediction App",
    page_icon="🩺",
    layout="centered"
)

# Load model and scaler helper
def load_artifacts(disease_name):
    prefix_map = {
        "Diabetes": "diabetes",
        "Heart Disease": "heart",
        "Breast Cancer": "breast_cancer"
    }
    prefix = prefix_map.get(disease_name)
    
    model_path = f"{prefix}_model.pkl"
    scaler_path = f"{prefix}_scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

def main():
    st.sidebar.title("Disease Prediction")
    selected_disease = st.sidebar.radio("Select Disease", ["Diabetes", "Heart Disease", "Breast Cancer"])
    
    st.title(f"🩺 {selected_disease} Prediction")
    
    model, scaler = load_artifacts(selected_disease)
    
    if model is None or scaler is None:
        st.warning(f"Model files for **{selected_disease}** not found.")
        st.info(f"Please train the model using 'train_all_models.py' and place the following files in this directory:\n- `{selected_disease.lower().replace(' ', '_')}_model.pkl`\n- `{selected_disease.lower().replace(' ', '_')}_scaler.pkl`")
        return

    st.write(f"Enter patient details below to predict the likelihood of {selected_disease}.")
    st.caption("Please ensure all values are within the specified medical ranges.")

    with st.form("prediction_form"):
        st.header("Patient Data")
        
        input_data = []
        
        if selected_disease == "Diabetes":
            col1, col2 = st.columns(2)
            with col1:
                # Ranges based on Pima dataset distributions and general medical knowledge
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, help="Number of times pregnant")
                glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120, help="Plasma glucose concentration (mg/dL)")
                blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, help="Diastolic blood pressure (mm Hg)")
                skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, help="Triceps skin fold thickness (mm)")
            with col2:
                insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=79, help="2-Hour serum insulin (mu U/ml)")
                bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0, step=0.1, help="Body mass index (weight in kg/(height in m)^2)")
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, help="Diabetes pedigree function")
                age = st.number_input("Age", min_value=0, max_value=120, value=33, help="Age in years")
            
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
            
        elif selected_disease == "Heart Disease":
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=50)
                sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
                cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                                format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
                trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120, help="Resting blood pressure (mm Hg)")
                chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200, help="Serum cholestoral in mg/dl")
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "True" if x == 1 else "False")
                restecg = st.selectbox("Resting ECG", [0, 1, 2], 
                                     format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x])
            with col2:
                thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=150)
                exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="ST depression induced by exercise relative to rest")
                slope = st.selectbox("Slope", [0, 1, 2], 
                                   format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
                ca = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4], help="Number of major vessels colored by flourosopy")
                thal = st.selectbox("Thalassemia", [0, 1, 2, 3], 
                                  format_func=lambda x: ["Normal", "Fixed Defect", "Reversable Defect", "Unknown"][x] if x < 3 else "Unknown")
            
            input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            
        elif selected_disease == "Breast Cancer":
            st.info("Please enter the mean values from the diagnosis report.")
            cols = st.columns(3)
            
            # Feature configuration with realistic ranges (approximate bounds for Breast Cancer Wisconsin dataset)
            # Typically: Radius (6-30), Texture (9-40), Perimeter (43-190), Area (143-2500)
            # Smoothness (0.05-0.16), Compactness (0.02-0.35), Concavity (0-0.43), Concave points (0-0.2)
            # Symmetry (0.1-0.3), Fractal dim (0.05-0.1)
            
            feature_config = [
                ("Radius Mean", 6.0, 30.0, 14.0), ("Texture Mean", 9.0, 40.0, 19.0), ("Perimeter Mean", 43.0, 190.0, 90.0),
                ("Area Mean", 143.0, 2500.0, 650.0), ("Smoothness Mean", 0.05, 0.25, 0.1), ("Compactness Mean", 0.01, 0.35, 0.1),
                ("Concavity Mean", 0.0, 0.45, 0.09), ("Concave Points Mean", 0.0, 0.20, 0.05), ("Symmetry Mean", 0.1, 0.35, 0.18),
                ("Fractal Dimension Mean", 0.04, 0.1, 0.06),
                # SE features typically smaller
                ("Radius SE", 0.1, 3.0, 0.4), ("Texture SE", 0.3, 5.0, 1.2), ("Perimeter SE", 0.7, 22.0, 2.8),
                ("Area SE", 6.0, 550.0, 40.0), ("Smoothness SE", 0.0, 0.04, 0.007), ("Compactness SE", 0.0, 0.15, 0.025),
                ("Concavity SE", 0.0, 0.4, 0.03), ("Concave Points SE", 0.0, 0.06, 0.01), ("Symmetry SE", 0.0, 0.08, 0.02),
                ("Fractal Dimension SE", 0.0, 0.03, 0.003),
                # Worst features typically larger than mean
                ("Radius Worst", 7.0, 40.0, 16.0), ("Texture Worst", 12.0, 50.0, 25.0), ("Perimeter Worst", 50.0, 260.0, 107.0),
                ("Area Worst", 185.0, 4300.0, 880.0), ("Smoothness Worst", 0.07, 0.23, 0.13), ("Compactness Worst", 0.02, 1.1, 0.25),
                ("Concavity Worst", 0.0, 1.3, 0.27), ("Concave Points Worst", 0.0, 0.3, 0.11), ("Symmetry Worst", 0.15, 0.7, 0.29),
                ("Fractal Dimension Worst", 0.05, 0.21, 0.08)
            ]
            
            input_data = []
            for i, (label, min_val, max_val, default) in enumerate(feature_config):
                with cols[i % 3]:
                    val = st.number_input(
                        label, 
                        min_value=float(min_val), 
                        max_value=float(max_val), 
                        value=float(default),
                        step=0.01,
                        format="%.4f"
                    )
                    input_data.append(val)

        submit_btn = st.form_submit_button("Predict")

    if submit_btn:
        input_array = np.array([input_data])
        
        try:
            scaled_input = scaler.transform(input_array)
            prediction = model.predict(scaled_input)
            prediction_proba = model.predict_proba(scaled_input)
            
            st.markdown("---")
            st.subheader("Prediction Result")
            
            if selected_disease == "Diabetes":
                is_positive = prediction[0] == 1
                pos_label = "Positive (Diabetes Risk)"
                neg_label = "Negative (Healthy)"
            elif selected_disease == "Heart Disease":
                is_positive = prediction[0] == 1
                pos_label = "Positive (Heart Disease Risk)"
                neg_label = "Negative (Healthy)"
            elif selected_disease == "Breast Cancer":
                is_positive = prediction[0] == 1
                pos_label = "Malignant (Cancerous)"
                neg_label = "Benign (Safe)"
                
            prob = prediction_proba[0][1] if is_positive else prediction_proba[0][0]
            
            if is_positive:
                st.error(f"**{pos_label}**")
                st.write(f"Confidence Level: **{prob:.2%}**")
            else:
                st.success(f"**{neg_label}**")
                st.write(f"Confidence Level: **{prob:.2%}**")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()
