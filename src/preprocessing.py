from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Membagi data menjadi set pelatihan dan pengujian.
    
    Args:
        X (pd.DataFrame): Fitur
        y (pd.Series): Target
        test_size (float): Proporsi data tes
        random_state (int): Seed untuk reproduktibilitas
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Data dibagi: {len(y_train)} data latih, {len(y_test)} data tes.")
    return X_train, X_test, y_train, y_test