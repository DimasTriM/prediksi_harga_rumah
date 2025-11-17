import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- Blok Konfigurasi Model (Sama seperti sebelumnya) ---
FITUR_COLS = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'knn_regressor_pipeline.joblib')

@st.cache_resource
def load_model():
    """Memuat pipeline model KNN yang sudah dilatih."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"File model TIDAK DITEMUKAN. Aplikasi mencari di path ini: {MODEL_PATH}")
        st.info("Pastikan file 'knn_regressor_pipeline.joblib' ada di dalam folder 'models' Anda.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.stop()
# --- Akhir Blok Konfigurasi Model ---


# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide"
)

# Memuat model
model_pipeline = load_model()

# --- Sidebar (Sama seperti sebelumnya) ---
with st.sidebar:
    st.header('Parameter Input Fitur')
    
    input_data = {}
    input_data['MedInc'] = st.slider('Pendapatan Median (x $10.000)', 0.5, 15.0, 3.87, 0.1)
    input_data['HouseAge'] = st.slider('Usia Rata-rata Rumah', 1, 52, 29, 1)
    input_data['AveRooms'] = st.slider('Rata-rata Kamar', 1.0, 15.0, 5.4, 0.1)
    input_data['AveBedrms'] = st.slider('Rata-rata Kamar Tidur', 0.5, 8.0, 1.1, 0.05)
    input_data['Population'] = st.number_input('Populasi Distrik', 3, 40000, 1425, 10)
    input_data['AveOccup'] = st.slider('Rata-rata Penghuni', 0.5, 15.0, 3.0, 0.1)

    st.subheader("Lokasi Geografis:")
    input_data['Latitude'] = st.slider('Garis Lintang (Latitude)', 32.0, 42.0, 35.6, 0.1)
    input_data['Longitude'] = st.slider('Garis Bujur (Longitude)', -125.0, -114.0, -119.5, 0.1)
    
    predict_button = st.button('🚀 Prediksi Harga', use_container_width=True)


# --- Halaman Utama (Tampilan JAUH LEBIH BAIK) ---

# 1. Judul dan Header
st.title('🏠 Aplikasi Prediksi Harga Rumah')
st.write("Selamat datang di aplikasi prediksi harga rumah California. Aplikasi ini menggunakan model Machine Learning (KNN) untuk mengestimasi harga rumah.")

# 2. Gambar Header (Ukuran Diperbaiki dan Rata Kiri)
# Kita hapus kolom dan ganti dengan parameter 'width'
st.image(
    "https://images.pexels.com/photos/1396122/pexels-photo-1396122.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
    caption="Ilustrasi properti modern",
    width=200  # <-- KITA ATUR LEBARNYA DI SINI (dalam piksel)
)

# 3. Logika Tampilan (Saat Tombol Ditekan atau Belum)

if predict_button:
    # --- TAMPILAN SETELAH PREDIKSI ---
    
    input_list = [input_data[col] for col in FITUR_COLS]
    input_df = pd.DataFrame([input_list], columns=FITUR_COLS)
    
    try:
        prediction = model_pipeline.predict(input_df)
        pred_value = prediction[0]
        harga_usd = pred_value * 100000
        
        st.header('🎉 Hasil Prediksi Anda')

        # --- KARTU HASIL PREDIKSI (DIBUAT LEBIH JELAS) ---
        with st.container(border=True):
            st.subheader("Estimasi Harga:")
            # Menampilkan hasil prediksi dengan format mata uang yang jelas
            st.metric(label="Prediksi Harga Rumah", value=f"${harga_usd:,.2f} USD")
            st.caption("Prediksi ini dihasilkan oleh model K-Nearest Neighbors (KNN) Regressor. "
                       "Harap diingat ini adalah estimasi berdasarkan data historis.")

        st.divider() # Garis pemisah

        st.subheader("Data Input yang Digunakan:")
        st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

else:
    # --- TAMPILAN AWAL (SEKARANG DENGAN KOLOM) ---
    st.info('ℹ️ Silakan masukkan semua nilai fitur di sidebar kiri dan klik tombol "Prediksi Harga".', icon="👈")
    
    st.divider()
    
    # Menggunakan kolom untuk tata letak yang lebih baik
    col1, col2 = st.columns([1.5, 1]) # Kolom teks lebih lebar
    
    with col1:
        st.header("Tentang Proyek Ini")
        st.write("""
        Proyek ini dibangun sebagai demonstrasi alur kerja *machine learning* end-to-end:
        
        1.  **Pengumpulan Data:** Menggunakan dataset California Housing dari Scikit-learn.
        2.  **Eksplorasi Data (EDA):** Menganalisis fitur dan korelasi di Jupyter Notebook.
        3.  **Pelatihan Model:** Melatih beberapa model regresi (Linear, Decision Tree, KNN).
        4.  **Evaluasi:** Model KNN (K-Nearest Neighbors) dipilih sebagai model terbaik (R-squared ~0.67).
        5.  **Deployment:** Aplikasi Streamlit ini memuat model KNN yang sudah dilatih (`.joblib`) untuk membuat prediksi *real-time*.
        """)
        st.write("Coba ubah-ubah nilai di sidebar untuk melihat bagaimana fitur yang berbeda memengaruhi prediksi harga!")

    with col2:
        # Menambahkan gambar pendukung
        st.image(
            "https://images.pexels.com/photos/3184418/pexels-photo-3184418.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1", 
            caption="Analisis Data dan Kolaborasi",
            use_container_width=True
        )