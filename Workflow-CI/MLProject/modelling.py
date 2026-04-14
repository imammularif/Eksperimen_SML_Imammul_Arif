import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def train_final_model():
    # Aktifkan logging otomatis MLflow
    mlflow.sklearn.autolog()

    # Memuat data hasil preprocessing
    # Pastikan file CSV ini ada di folder yang sama dengan modelling.py
    df = pd.read_csv('heart_disease_preprocessed.csv')
    X = df.drop('target', axis=1)
    y = (df['target'] > 0).astype(int)
    
    # Mulai sesi pencatatan MLflow
    with mlflow.start_run(run_name="Heart_Disease_Model_Training"):
        # Pelatihan
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Simpan model secara lokal
        joblib.dump(model, 'heart_model.joblib')
        
        print("Model Berhasil Dilatih dan dicatat ke MLflow!")

if __name__ == "__main__":
    train_final_model()