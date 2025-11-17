import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name="model"):
    """
    Membuat pipeline (Scaler + Model), melatih, mengevaluasi,
    dan menyimpan model.
    
    Args:
        model: Objek model Scikit-learn (misal: LinearRegression())
        X_train, y_train: Data latih
        X_test, y_test: Data tes
        model_name (str): Nama model untuk penamaan file
        
    Returns:
        tuple: (pipeline, mse, r2)
    """
    
    # 1. Membuat Pipeline
    # Penting: Scaling data (StandardScaler) sangat penting untuk
    # Linear Regression dan KNN.
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),  # Langkah 1: Scaling data
        ('model', model)              # Langkah 2: Model
    ])
    
    # 2. Melatih pipeline
    print(f"Melatih model {model_name}...")
    pipeline.fit(X_train, y_train)
    
    # 3. Memprediksi data tes
    y_pred = pipeline.predict(X_test)
    
    # 4. Mengevaluasi model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"--- Hasil untuk {model_name} ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    
    # 5. Menyimpan model
    # Kita simpan di folder 'models' di luar folder 'src'
    model_filename = f"../models/{model_name.lower().replace(' ', '_')}_pipeline.joblib"
    joblib.dump(pipeline, model_filename)
    print(f"Model disimpan di: {model_filename}\n")
    
    return pipeline, mse, r2