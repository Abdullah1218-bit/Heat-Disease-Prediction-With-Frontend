import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and preprocessing objects
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
cols_for_scaling = joblib.load('cols_for_scaling.pkl')
categoric_cols = joblib.load('categoric_cols.pkl')
cols_with_outliers = joblib.load('cols_with_outliers.pkl')
label_encoders = joblib.load('label_encoders.pkl') 

# Streamlit App Title
st.title('❤️ Heart Disease Prediction')
st.markdown('Provide the following details to predict the risk of heart disease:')

# Input Fields
age = st.slider('Age', 18, 100, 30)
gender = st.selectbox('Gender', ['M', 'F'])
chest_pain_type = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', 90, 200, 120)
cholesterol = st.number_input('Cholesterol (mg/dL)', 0, 600, 180)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', [0, 1])
resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_hr = st.number_input('Maximum Heart Rate Achieved (bpm)', 60, 220, 150)
exercise_angina = st.selectbox('Exercise-Induced Angina', ['N', 'Y'])
oldpeak = st.number_input('Oldpeak (ST depression)', 0.0, 10.0, 1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

# On Predict Button Click
if st.button('🧠 Predict'):

    # Create DataFrame from input
    input_data = {
        'Age': age,
        'Sex': gender,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical features using the saved LabelEncoders
    for col in categoric_cols:
        if col in input_df.columns:
            encoder = label_encoders.get(col)
            if encoder:
                value = input_df.at[0, col]
                if value in encoder.classes_:
                    input_df[col] = encoder.transform([value])
                else:
                    st.error(f"⚠️ Value '{value}' for column '{col}' was not seen during training. Please select a valid option.")
                    st.stop()


    # Apply scaling
    input_df[cols_for_scaling] = scaler.transform(input_df[cols_for_scaling])

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    # Display result
    if prediction == 1:
        st.error(f"❌ High Risk: The patient is likely to have heart disease.\n\n**Confidence: {probability:.2f}%**")
    else:
        st.success(f"✅ Low Risk: The patient is unlikely to have heart disease.\n\n**Confidence: {100 - probability:.2f}%**")
