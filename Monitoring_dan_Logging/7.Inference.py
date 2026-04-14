import joblib
import numpy as np

# Load model dan scaler dari folder Membangun_model
model = joblib.load('Membangun_model/heart_model.joblib')
scaler = joblib.load('Membangun_model/scaler.joblib')

# Simulasi data pasien baru (13 fitur sesuai urutan dataset)
# Usia, Jenis Kelamin, Tipe Nyeri Dada, Tekanan Darah, Kolesterol, dst.
data_pasien = np.array([[55, 1, 0, 130, 250, 0, 1, 150, 0, 1.2, 1, 0, 2]])

# 1. Preprocessing (Scaling)
data_scaled = scaler.transform(data_pasien)

# 2. Prediksi
prediksi = model.predict(data_scaled)
probabilitas = model.predict_proba(data_scaled)

# 3. Hasil
status = "Risiko Penyakit Jantung" if prediksi[0] == 1 else "Jantung Sehat"
print(f"--- HASIL INFERENCE ---")
print(f"Status: {status}")
print(f"Tingkat Keyakinan: {probabilitas[0][prediksi[0]]*100:.2f}%")