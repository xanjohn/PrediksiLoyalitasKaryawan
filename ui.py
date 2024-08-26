import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# # Load the pre-trained model
# model = joblib.load('model.pkl')  # Ganti dengan path ke model Anda

# # Function to encode categorical features
# def encode_features(df):
#     le = LabelEncoder()
#     categorical_cols = ['Department', 'Gender', 'MaritalStatus', 'OverTime']
#     for col in categorical_cols:
#         df[col] = le.fit_transform(df[col])
#     return df

# Function to make predictions
def predict(attrs):
    df = pd.DataFrame([attrs], columns=[
        'Age', 'Department', 'DistanceFromHome', 'Education',
        'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
        'JobLevel', 'JobSatisfaction', 'MaritalStatus', 'MonthlyRate',
        'OverTime', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'TotalWorkingYears', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ])
    df = encode_features(df)
    prediction = model.predict(df)
    return prediction[0]

st.title('Prediksi Resign Pegawai')

# Input fields
age = st.slider('Age', 18, 65, 30)
department = st.selectbox('Department', ['Sales', 'HR', 'Development', 'Finance', 'Marketing'])
distance_from_home = st.slider('DistanceFromHome', 1, 30, 10)
education = st.selectbox('Education', [1, 2, 3, 4])  # Asumsi kualifikasi pendidikan sebagai angka
environment_satisfaction = st.slider('EnvironmentSatisfaction', 1, 4, 3)
gender = st.selectbox('Gender', ['Male', 'Female'])
hourly_rate = st.slider('HourlyRate', 10, 100, 20)
job_involvement = st.slider('JobInvolvement', 1, 4, 3)
job_level = st.slider('JobLevel', 1, 5, 3)
job_satisfaction = st.slider('JobSatisfaction', 1, 4, 3)
marital_status = st.selectbox('MaritalStatus', ['Single', 'Married', 'Divorced'])
monthly_rate = st.slider('MonthlyRate', 1000, 50000, 3000)
over_time = st.selectbox('OverTime', ['Yes', 'No'])
percent_salary_hike = st.slider('PercentSalaryHike', 1, 30, 5)
performance_rating = st.slider('PerformanceRating', 1, 4, 3)
relationship_satisfaction = st.slider('RelationshipSatisfaction', 1, 4, 3)
total_working_years = st.slider('TotalWorkingYears', 0, 40, 5)
work_life_balance = st.slider('WorkLifeBalance', 1, 4, 3)
years_at_company = st.slider('YearsAtCompany', 0, 20, 5)
years_in_current_role = st.slider('YearsInCurrentRole', 0, 10, 3)
years_since_last_promotion = st.slider('YearsSinceLastPromotion', 0, 10, 2)
years_with_curr_manager = st.slider('YearsWithCurrManager', 0, 20, 5)

# Predict button
if st.button('Predict'):
    attrs = [
        age, department, distance_from_home, education,
        environment_satisfaction, gender, hourly_rate, job_involvement,
        job_level, job_satisfaction, marital_status, monthly_rate,
        over_time, percent_salary_hike, performance_rating,
        relationship_satisfaction, total_working_years, work_life_balance,
        years_at_company, years_in_current_role, years_since_last_promotion,
        years_with_curr_manager
    ]
    result = predict(attrs)
    if result == 1:
        st.write('Prediksi: Karyawan kemungkinan akan keluar.')
    else:
        st.write('Prediksi: Karyawan kemungkinan akan tetap.')
