import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

# Load the pre-trained model
model = joblib.load('model2.pkl')  # Ganti dengan path ke model Anda

# Function to encode categorical features
def encode_features(df):
    le = LabelEncoder()
    categorical_cols = ['status_perkawinan', 'gaji']  # Tambahkan kolom kategorikal lain jika diperlukan
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df

# Function to make predictions and get probability
def predict(attrs):
    df = pd.DataFrame([attrs], columns=[
        'satisfaction_level', 'last_evaluation', 'number_project',
        'average_montly_hours', 'time_spend_company', 'Work_accident',
        'promotion_last_5years', 'Hubungan_dengan_atasan', 'performa', 'gaji',
        'beban_kerja', 'apresiasi_atasan', 'jam_kerja', 'umur',
        'jarak_dari_kantor', 'status_perkawinan', 'jumlah_anak',
        'riwayat_penyakit'
    ])
    df = encode_features(df)
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]  # Probabilitas untuk kelas 'keluar'
    return prediction[0], probability[0]

st.title('Prediksi Keluar Karyawan dengan Probabilitas')

# Input fields
satisfaction_level = st.slider('Tingkat Kepuasan', 0.0, 1.0, 0.5)
last_evaluation = st.slider('Evaluasi Terakhir', 0.0, 1.0, 0.5)
number_project = st.slider('Jumlah Project Yang Dikerjakan', 1, 10, 3)
average_montly_hours = st.slider('Rata-Rata Jam Kerja Perbulan', 80, 320, 160)
time_spend_company = st.slider('Waktu Yang Dihabiskan Di Perusahaan (Tahun)', 0, 20, 3)
Work_accident = st.selectbox('Kejadian Saat Kerja', [0, 1])
promotion_last_5years = st.selectbox('Promosi Dalam 5 tahun Terakhir', [0, 1])
Hubungan_dengan_atasan = st.slider('Hubungan dengan Atasan', 0.0, 1.0, 0.5)
performa = st.slider('Performa', 0.0, 1.0, 0.5)
gaji = st.selectbox('Gaji', ['Rendah', 'Sedang', 'Tinggi'])
beban_kerja = st.slider('Beban Kerja', 0.0, 1.0, 0.5)
apresiasi_atasan = st.slider('Apresiasi Atasan', 0.0, 1.0, 0.5)
jam_kerja = st.slider('Jam Kerja', 80, 320, 160)
umur = st.slider('Umur', 18, 65, 30)
jarak_dari_kantor = st.slider('Jarak dari Kantor (Km)', 0, 100, 10)
status_perkawinan = st.selectbox('Status Perkawinan', ['Single', 'Married', 'Divorced'])
jumlah_anak = st.slider('Jumlah Anak', 0, 10, 2)
riwayat_penyakit = st.selectbox('Riwayat Penyakit', [0, 1])

# Predict button
if st.button('Predict'):
    attrs = [
        satisfaction_level, last_evaluation, number_project,
        average_montly_hours, time_spend_company, Work_accident,
        promotion_last_5years, Hubungan_dengan_atasan, performa, gaji,
        beban_kerja, apresiasi_atasan, jam_kerja, umur,
        jarak_dari_kantor, status_perkawinan, jumlah_anak,
        riwayat_penyakit
    ]
    result, probability = predict(attrs)
    
    if result == 1:
        st.write('Prediksi: Karyawan kemungkinan akan keluar.')
    else:
        st.write('Prediksi: Karyawan kemungkinan akan bertahan.')

    
    # Visualize the probability using Pie Chart
    labels = ['Bertahan', 'Keluar']
    sizes = [1 - probability, probability]
    colors = ['green', 'red']
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)
