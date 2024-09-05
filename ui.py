import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px 

# Load the pre-trained model
model = joblib.load('model1.pkl')  # Ganti dengan path ke model Anda

# Function to encode categorical features
def encode_features(df):
    # Encode Gender, MaritalStatus, and OverTime as one-hot vectors
    df = pd.get_dummies(df, columns=['Gender', 'MaritalStatus', 'OverTime'], drop_first=True)
    return df

# Function to make predictions
def predict(attrs):
    df = pd.DataFrame([attrs], columns=[
        'Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
        'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
        'MonthlyRate', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'TotalWorkingYears', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager', 'Gender_Male', 'MaritalStatus_Married',
        'MaritalStatus_Single', 'OverTime_Yes'
    ])
    df = encode_features(df)
    prediction = model.predict(df)
    proba = model.predict_proba(df)  # Get prediction probabilities
    return prediction[0], proba

st.title('Prediksi Resign Pegawai')

# Input fields
age = st.slider('Age', 18, 65, 30)
distance_from_home = st.slider('DistanceFromHome', 1, 30, 10)
education = st.selectbox('Education', [1, 2, 3, 4])  # Education level as a number
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

# Convert inputs to the expected format for prediction
attrs = [
    age, distance_from_home, education, environment_satisfaction,
    hourly_rate, job_involvement, job_level, job_satisfaction,
    monthly_rate, percent_salary_hike, performance_rating,
    relationship_satisfaction, total_working_years, work_life_balance,
    years_at_company, years_in_current_role, years_since_last_promotion,
    years_with_curr_manager, 1 if gender == 'Male' else 0,
    1 if marital_status == 'Married' else 0,
    1 if marital_status == 'Single' else 0,
    1 if over_time == 'Yes' else 0
]

# Predict button
if st.button('Predict'):
    result, proba = predict(attrs)
    
    st.write(f'Prediksi: {"Karyawan kemungkinan akan keluar" if result == 1 else "Karyawan kemungkinan akan bertahan"}')
    
    # Display the probabilities
    prob_df = pd.DataFrame(proba, columns=['Tetap', 'Keluar'])
    st.write(prob_df)
    
    # Visualize probabilities
    fig = px.bar(prob_df.T, title='Probabilitas', labels={'value': 'Probabilitas', 'index': 'Label'})
    st.plotly_chart(fig)