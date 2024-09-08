import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

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

# Function to convert dataframe to excel or csv format
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Function to style the predictions DataFrame
def style_predictions(df):
    def highlight_cells(val):
        color = 'background-color: #d9f2d9' if val < 0.5 else 'background-color: #f9d6d5'
        return color

    styled_df = df.style.applymap(highlight_cells, subset=['Probabilitas Bertahan', 'Probabilitas Keluar'])
    return styled_df

#SETAYLE AWIKWOK
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    h1 {
        font-family: 'Poppins', sans-serif;
        font-style: bold;
        font-size: 30px;
        text-align: center;
        margin-bottom: 4px;
        color: black;
    }

    .justified-text {
        text-align: justify;
        font-size: 13px;
        line-height: 1.5;
        font-family: 'Poppins', sans-serif;
    }

    .input-label {
        font-size: 18px; /* Mengatur ukuran font label input */
        font-family: 'Poppins', sans-serif;
        font-weight: bold;
        margin-bottom: -5px;  /* Mengurangi jarak antara label dan input */
        display: block;
    }

    .stTextInput > div, .stSelectbox > div, .stSlider > div {
        margin-top: 10px; /* Menghapus margin atas pada elemen input */
    }

    .stSelectbox > label, .stTextInput > label {
        margin-top: -30px;
    }

    /* Tambahkan gaya untuk slider */
    .stSlider > div > div > div {
        border-radius: 10px;
        background-color: #f0f0f5; /* Warna latar belakang slider */
        padding: 8px; /* Padding slider */
        margin-top: -1px; /* Margin slider */
    }

    .prediction-result {
        padding: 10px;
        background-color: #f0f8ff;
        border-radius: 8px;
        border-left: 4px solid #4682b4;
        margin-bottom: 20px;
    }

    .prediction-result h2 {
        font-size: 24px;
        font-family: 'Poppins', sans-serif;
        color: #4682b4;
        margin: 0;
    }

    .prediction-detail {
        font-size: 18px;
        font-family: 'Poppins', sans-serif;
        margin-top: 8px;
    }

    .prediction-detail span {
        font-weight: bold;
        color: #000;
    }
    
    .bold-text {
        font-family: 'Poppins', sans-serif;
        font-weight: bold;
        font-size: 18px; /* Sesuaikan ukuran font jika perlu */
        color: black; /* Sesuaikan warna font jika perlu */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title('SI PALING HRD')

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
        <div class="justified-text">
        Aplikasi ini menggunakan machine learning untuk memprediksi kemungkinan seorang karyawan keluar dari perusahaan berdasarkan faktor-faktor seperti kepuasan kerja, jumlah proyek, jam kerja, dan hubungan dengan atasan. Selain memprediksi apakah karyawan akan bertahan atau keluar, aplikasi ini juga menampilkan peluang dari prediksi tersebut untuk membantu pengambilan keputusan manajemen.
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.image("./logo.jpg")

st.divider()

# Input fields
st.markdown('<div class="input-label">Nama Karyawan</div>', unsafe_allow_html=True)
name = st.text_input('', placeholder='Masukkan nama karyawan')

# Function to transform the 'Ada/Tidak Ada' back to 1/0 for prediction
def transform_to_binary(option):
    return 1 if option == 'Ada' else 0

# Using a placeholder in selectbox workaround
def selectbox_with_placeholder(label, options):
    st.markdown(f'<div class="input-label">{label}</div>', unsafe_allow_html=True)
    option = st.selectbox('', options, key=label)
    if option == 'Pilih satu pilihan':
        st.warning(f"Silahkan pilih opsi untuk {label}!")
    return option

# Updated selectbox calls with 'Ada' and 'Tidak Ada'
Work_accident = selectbox_with_placeholder('Kejadian Saat Kerja', ['Pilih satu pilihan', 'Tidak Ada', 'Ada'])
promotion_last_5years = selectbox_with_placeholder('Promosi Dalam 5 tahun Terakhir', ['Pilih satu pilihan', 'Tidak Ada', 'Ada'])
riwayat_penyakit = selectbox_with_placeholder('Riwayat Penyakit', ['Pilih satu pilihan', 'Tidak Ada', 'Ada'])

# Convert 'Ada/Tidak Ada' to 1/0
if Work_accident != 'Pilih satu pilihan':
    Work_accident = transform_to_binary(Work_accident)
if promotion_last_5years != 'Pilih satu pilihan':
    promotion_last_5years = transform_to_binary(promotion_last_5years)
if riwayat_penyakit != 'Pilih satu pilihan':
    riwayat_penyakit = transform_to_binary(riwayat_penyakit)

# Convert other selectbox options as needed
gaji = selectbox_with_placeholder('Gaji', ['Pilih satu pilihan', 'Rendah', 'Sedang', 'Tinggi'])

# Updated selectbox options for 'status_perkawinan'
status_perkawinan = selectbox_with_placeholder('Status Perkawinan', ['Pilih satu pilihan', 'Belum Kawin', 'Kawin', 'Cerai'])

# Menggunakan slider standar, tetapi menambahkan label kustom
def slider_with_label(label, min_val, max_val, default_val):
    st.markdown(f'<div class="input-label">{label}</div>', unsafe_allow_html=True)
    return st.slider('', min_val, max_val, default_val, key=label)

satisfaction_level = slider_with_label('Tingkat Kepuasan', 0.0, 1.0, 0.5)
last_evaluation = slider_with_label('Evaluasi Terakhir', 0.0, 1.0, 0.5)
number_project = slider_with_label('Jumlah Project Yang Dikerjakan', 1, 10, 3)
average_montly_hours = slider_with_label('Rata-Rata Jam Kerja Perbulan', 80, 320, 160)
time_spend_company = slider_with_label('Waktu Yang Dihabiskan Di Perusahaan (Tahun)', 0, 20, 3)
Hubungan_dengan_atasan = slider_with_label('Hubungan dengan Atasan', 0.0, 1.0, 0.5)
performa = slider_with_label('Performa', 0.0, 1.0, 0.5)
beban_kerja = slider_with_label('Beban Kerja', 0.0, 1.0, 0.5)
apresiasi_atasan = slider_with_label('Apresiasi Atasan', 0.0, 1.0, 0.5)
jam_kerja = slider_with_label('Jam Kerja', 80, 320, 160)
umur = slider_with_label('Umur', 18, 65, 30)
jarak_dari_kantor = slider_with_label('Jarak dari Kantor (Km)', 0, 100, 10)
jumlah_anak = slider_with_label('Jumlah Anak', 0, 10, 2)

# Initialize the dataset to store predictions
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = pd.DataFrame(columns=['Nama', 'Prediksi', 'Probabilitas Bertahan', 'Probabilitas Keluar'])

# Predict button
if st.button('Predict'):
    if name:  # Ensure name is provided
        # Check if all selectbox options are valid
        if Work_accident != 'Pilih satu pilihan' and promotion_last_5years != 'Pilih satu pilihan' and gaji != 'Pilih satu pilihan' and status_perkawinan != 'Pilih satu pilihan' and riwayat_penyakit != 'Pilih satu pilihan':
            attrs = [
                satisfaction_level, last_evaluation, number_project,
                average_montly_hours, time_spend_company, Work_accident,
                promotion_last_5years, Hubungan_dengan_atasan, performa, gaji,
                beban_kerja, apresiasi_atasan, jam_kerja, umur,
                jarak_dari_kantor, status_perkawinan, jumlah_anak,
                riwayat_penyakit
            ]
            result, probability = predict(attrs)
            # Modify the result display with HTML and CSS
            if result == 1:
                st.markdown(
                    f"""
                    <div class="prediction-result">
                        <h2>{name} kemungkinan akan keluar.</h2>
                        <div class="prediction-detail">Probabilitas Keluar : <span>{probability:.2f}</span></div>
                        <div class="prediction-detail">Probabilitas Bertahan : <span>{1 - probability:.2f}</span></div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="prediction-result">
                        <h2>{name} kemungkinan akan bertahan.</h2>
                        <div class="prediction-detail">Probabilitas Keluar : <span>{probability:.2f}</span></div>
                        <div class="prediction-detail">Probabilitas Bertahan: <span>{1 - probability:.2f}</span></div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

            # Add result to the dataframe
            new_row = pd.DataFrame({
                'Nama': [name],
                'Prediksi': ['Keluar' if result == 1 else 'Bertahan'],
                'Probabilitas Bertahan': [1 - probability],
                'Probabilitas Keluar': [probability]
            })
            st.session_state['predictions'] = pd.concat([st.session_state['predictions'], new_row], ignore_index=True)

            # Visualize the probability using Pie Chart
            labels = ['Bertahan', 'Keluar']
            sizes = [1 - probability, probability]
            colors = ['green', 'red']

            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            # Display the styled DataFrame below the pie chart
            st.markdown("<div class='bold-text'>Dataset Hasil Prediksi:</div>", unsafe_allow_html=True)
            styled_df = style_predictions(st.session_state['predictions'])
            st.dataframe(styled_df, use_container_width=True)

            # Download buttons
            excel_data = to_excel(st.session_state['predictions'])
            csv_data = to_csv(st.session_state['predictions'])

            st.download_button(label='Download Excel', data=excel_data, file_name='predictions.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            st.download_button(label='Download CSV', data=csv_data, file_name='predictions.csv', mime='text/csv')
        else:
            st.warning("Silahkan pilih semua opsi yang diperlukan!")
    else:
        st.warning("Nama harus diisi untuk melakukan prediksi!")