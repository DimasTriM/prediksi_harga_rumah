import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_data():
    """
    Memuat dataset California Housing Prices dari Scikit-learn
    dan mengembalikannya sebagai fitur (X) dan target (y).
    
    Returns:
        X (pd.DataFrame): Fitur-fitur dataset
        y (pd.Series): Target dataset (MedHouseVal)
    """
    # Mengambil data
    data_housing = fetch_california_housing()
    
    # Membuat DataFrame untuk fitur
    X = pd.DataFrame(data_housing.data, columns=data_housing.feature_names)
    
    # Membuat Series untuk target
    y = pd.Series(data_housing.target, name='MedHouseVal')
    
    print("Dataset berhasil dimuat.")
    print(f"Jumlah baris: {len(X)}, Jumlah fitur: {len(X.columns)}")
    
    return X, y

if __name__ == '__main__':
    # Uji coba cepat untuk memastikan fungsi berjalan
    X_data, y_data = load_data()
    print("\nContoh Fitur (X):")
    print(X_data.head())
    print("\nContoh Target (y):")
    print(y_data.head())