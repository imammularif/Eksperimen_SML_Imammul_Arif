import pandas as pd
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Setup Logging sederhana
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def train_with_tuning():
    logging.info("Memulai proses Hyperparameter Tuning...")
    
    # 1. Load data hasil preprocessing
    # Pastikan file ini ada di folder yang sama
    df = pd.read_csv('heart_disease_preprocessed.csv')
    X = df.drop('target', axis=1)
    y = (df['target'] > 0).astype(int)
    
    # 2. Inisialisasi Model
    rf = RandomForestClassifier(random_state=42)
    
    # 3. Tentukan Parameter yang mau dicoba (Tuning)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    # 4. Pencarian Parameter Terbaik dengan Grid Search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    logging.info(f"Parameter terbaik ditemukan: {grid_search.best_params_}")
    
    # 5. Simpan Model Terbaik
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'heart_model_tuned.joblib')
    
    logging.info("Model hasil tuning berhasil disimpan sebagai 'heart_model_tuned.joblib'")

if __name__ == "__main__":
    train_with_tuning()