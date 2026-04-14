from prometheus_client import start_http_server, Gauge
import time
import random

# Membuat metrik monitoring
acc_gauge = Gauge('model_accuracy', 'Akurasi model jantung')

if __name__ == '__main__':
    # Jalankan server di port 8000
    start_http_server(8000)
    print("Exporter berjalan. Buka http://localhost:8000 untuk melihat metrik.")
    
    while True:
        # Simulasi akurasi yang naik turun antara 0.80 sampai 0.95
        acc_gauge.set(random.uniform(0.80, 0.95))
        time.sleep(10)