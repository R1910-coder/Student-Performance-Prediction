import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load saved files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("🎓 Student Performance Prediction System")

st.write("Enter student details:")

# User inputs (basic important ones)
hours = st.number_input("Hours Studied", 0, 24)
attendance = st.number_input("Attendance", 0, 100)
sleep = st.number_input("Sleep Hours", 0, 12)
previous = st.number_input("Previous Score", 0, 100)
tutoring = st.number_input("Tutoring Sessions", 0, 10)
physical = st.number_input("Physical Activity", 0, 10)

# Create input dataframe
input_dict = {
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Sleep_Hours": sleep,
    "Previous_Scores": previous,
    "Tutoring_Sessions": tutoring,
    "Physical_Activity": physical
}

input_df = pd.DataFrame([input_dict])

# Convert to full feature format
input_df = pd.get_dummies(input_df)

# Match training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# Prediction
if st.button("Predict Score"):
    prediction = model.predict(input_df)[0]
    st.success(f"📊 Predicted Exam Score: {prediction:.2f}")