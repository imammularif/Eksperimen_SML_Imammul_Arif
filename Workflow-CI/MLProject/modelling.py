import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os

# 1. Mendapatkan path folder saat ini agar tracking tidak nyasar
current_dir = os.getcwd()
mlflow.set_tracking_uri(f"file:///{current_dir}/mlruns")

# 2. Set nama eksperimen (Ini yang akan muncul di sisi kiri UI)
# Pakai nama unik agar tidak tertumpuk di 'Default'
mlflow.set_experiment("Eksperimen_Imammul_Arif")

# 3. Aktifkan Autolog
mlflow.autolog()

# Load Data
df = pd.read_csv('heart_disease_preprocessed.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Jalankan Training
with mlflow.start_run(run_name="Run_Final_Submission"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # --- BAGIAN KRUSIAL ---
    # Memaksa simpan folder 'model' yang berisi model.pkl dan MLmodel
    mlflow.sklearn.log_model(sk_model=rf, artifact_path="model") 
    
    # Metrik manual untuk syarat kriteria 'Skilled'
    accuracy = rf.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    print("-" * 30)
    print(f"Selesai! Eksperimen tersimpan di folder: {current_dir}/mlruns")
    print("-" * 30)