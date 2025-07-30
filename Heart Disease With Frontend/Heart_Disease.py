# Heart Disease Prediction Model Training Script

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (adjust path if needed)
df = pd.read_csv("heart.csv")  # Replace with your dataset file name

# Handle outliers using RobustScaler
cols_with_outliers = ['RestingBP', 'Cholesterol']
robust_scaler = RobustScaler()
df[cols_with_outliers] = robust_scaler.fit_transform(df[cols_with_outliers])

# Define categorical columns and their expected values
categoric_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
label_values = {
    'Sex': ['F', 'M'],
    'ChestPainType': ['ATA', 'NAP', 'TA', 'ASY'],
    'RestingECG': ['Normal', 'ST', 'LVH'],
    'ExerciseAngina': ['N', 'Y'],
    'ST_Slope': ['Up', 'Flat', 'Down']
}

label_encoders = {}

# Encode categorical features using expected full values
for col in categoric_cols:
    le = LabelEncoder()
    le.fit(label_values[col])  # Ensure all expected categories are learned
    df[col] = le.transform(df[col])
    label_encoders[col] = le

# Save encoders and column info
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(categoric_cols, 'categoric_cols.pkl')
joblib.dump(cols_with_outliers, 'cols_with_outliers.pkl')

# Scale features including encoded categoricals
cols_for_scaling = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
df[cols_for_scaling] = scaler.fit_transform(df[cols_for_scaling])

# Save scaler and scaling columns
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(cols_for_scaling, 'cols_for_scaling.pkl')

# Split data
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = CatBoostClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'heart_disease_model.pkl')