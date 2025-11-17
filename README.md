# Proyek: Prediksi Harga Rumah (California Housing)

Proyek ini adalah latihan machine learning untuk memprediksi harga median rumah di distrik-distrik California berdasarkan berbagai fitur.

## 🎯 Tugas
Regresi: Memprediksi nilai numerik (Median House Value).

## 💾 Dataset
Dataset yang digunakan adalah **California Housing Prices** dari Scikit-learn.

**Fitur (X):**
* `MedInc`: Pendapatan median
* `HouseAge`: Usia rata-rata rumah
* `AveRooms`: Rata-rata jumlah kamar
* `AveBedrms`: Rata-rata jumlah kamar tidur
* `Population`: Populasi distrik
* `AveOccup`: Rata-rata penghuni
* `Latitude`: Garis lintang
* `Longitude`: Garis bujur

**Target (y):**
* `MedHouseVal`: Harga median rumah (dalam ratusan ribu USD)

## 🤖 Model yang Digunakan
1.  **Linear Regression**: Model dasar sebagai *baseline*.
2.  **Decision Tree Regressor**: Model non-linear.
3.  **K-Nearest Neighbors (KNN) Regressor**: Model berbasis jarak.

## ⚙️ Struktur Proyek
```
Prediksi_Harga_Rumah/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 1_explorasi_data.ipynb
│   └── 2_eksperimen_model.ipynb
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── model_training.py
└── models/
    └── (Folder ini berisi model .joblib yang disimpan)
```

## 🚀 Cara Menjalankan
1.  Pastikan Anda memiliki Python dan telah menginstal semua *requirements*:
    ```bash
    pip install -r requirements.txt
    ```
2.  Jalankan skrip `main.py` dari dalam folder `src/`:
    ```bash
    cd src
    python main.py
    ```
3.  Hasil evaluasi model akan dicetak di konsol, dan model yang telah dilatih (dalam bentuk pipeline) akan disimpan di folder `models/`.