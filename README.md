❤️ Heart Disease Prediction Web App
This repository contains a machine learning web application for Heart Disease Prediction built using Streamlit and trained with the CatBoostClassifier.

📁 Project Structure
There are two main files:

1) Heart_Disease.py

Handles data preprocessing, model training, and saving.

2) Frontend.py

Builds the Streamlit web interface for user interaction and predictions.

🧠 Heart_Disease.py – Model Building

📊 Dataset

Dataset name: heartdisease.csv

Source:https://www.kaggle.com/code/siamhosaain/heart-disease

🔧 Key Features and Workflow

Applied RobustScaler on RestingBP and Cholesterol to handle outliers.

Defined all categorical columns and their expected values.

Performed manual encoding using expected full-value mappings.

Saved label encoders and column metadata using joblib.

Scaled all features (including encoded categoricals) with RobustScaler.

Saved the scaler and the list of scaled columns.

Split the dataset into training and testing sets.

Trained a CatBoostClassifier model.

Saved the trained model using joblib.

💻 Frontend.py – Streamlit App

This is a Streamlit web app that allows users to input their health data and get heart disease risk predictions.

📝 User Input Form Includes:

Age

Gender

Chest pain type

Resting blood pressure

Cholesterol

Fasting blood sugar

Resting ECG

Maximum heart rate

Exercise-induced angina

ST depression (Oldpeak)

ST slope

🚀 Prediction Workflow

Loads the trained model and preprocessing tools:

Scaler

Label encoders

Column lists

Converts user input into a DataFrame.

Encodes categorical values using saved label encoders.

Scales numerical features using the saved scaler.

Predicts heart disease risk using the trained model.

Displays the result:

❌ High Risk: with prediction confidence.

✅ Low Risk: with prediction confidence.
