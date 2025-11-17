import os
from data_loader import load_data
from preprocessing import split_data
from model_training import train_and_evaluate

# Import model-model yang akan digunakan
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

def main():
    """
    Fungsi utama untuk menjalankan alur kerja ML.
    """
    print("--- Memulai Alur Kerja Prediksi Harga Rumah ---")
    
    # Pastikan folder 'models' ada
    # (Kita berada di 'src', jadi kita cek '../models')
    models_dir = '../models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Folder '{models_dir}' dibuat.")

    # 1. Memuat Data
    print("\n1. Memuat Data...")
    X, y = load_data()
    
    # 2. Membagi Data
    print("\n2. Membagi Data...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    # 3. Melatih dan Mengevaluasi Model
    print("\n3. Melatih dan Mengevaluasi Model...")
    
    # Definisikan model-model yang akan dilatih
    models_to_train = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "KNN Regressor": KNeighborsRegressor() # Menggunakan K=5 (default)
    }
    
    # Latih dan evaluasi setiap model
    for name, model in models_to_train.items():
        train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name=name)
        
    print("--- Alur Kerja Selesai ---")

if __name__ == "__main__":
    main()