# 🚚 Delivery Delay Prediction - End-to-End MLOps Pipeline

Proyek ini bertujuan untuk memprediksi potensi keterlambatan pengiriman logistik berdasarkan data historis operasional. Kami menggunakan pendekatan **Deep Learning (TensorFlow)** yang diintegrasikan ke dalam arsitektur penyajian model (**Model Serving**) menggunakan **FastAPI** dan **Streamlit**.

## 📊 Ringkasan Analisis & Temuan Bisnis (Business Insights)

1. **Tingkat Keterlambatan Pengiriman (26.68%)**: Rata-rata 1 dari 4 pengiriman mengalami keterlambatan. Ini merupakan angka yang cukup besar dan berdampak negatif pada retensi pelanggan.
2. **Dampak Cuaca Ekstrem**: Cuaca badai (**Stormy**) meningkatkan kemungkinan delay hingga **41.45%**, diikuti oleh hujan (**Rainy**) sebesar **37.35%**. Disarankan untuk menerapkan penyesuaian waktu estimasi dinamis (*buffer time* otomatis).
3. **Anomali Layanan Premium (Express)**: Pengiriman mode **Express** memiliki tingkat keterlambatan mencapai **73.78%**! Hal ini menunjukkan target waktu pengantaran saat ini tidak realistis atau kapasitas kurir tidak memadai di lapangan.
4. **Kepuasan Pelanggan**: Keterlambatan memiliki korelasi negatif yang sangat kuat (**-0.78**) dengan `delivery_rating`, mengonfirmasi bahwa keterlambatan merupakan penyebab utama turunnya kepuasan pelanggan.

---

## 🛠️ Arsitektur MLOps & Serving

Proyek ini memisahkan antara bagian antarmuka pengguna (Frontend) dengan server komputasi model (Backend) untuk mensimulasikan lingkungan produksi nyata:

```
                  ┌──────────────────────┐
                  │   Streamlit Client   │ (Port 8501)
                  └──────────┬───────────┘
                             │
                     POST /predict (HTTP)
                             │
                             ▼
                  ┌──────────────────────┐
                  │    FastAPI Server    │ (Port 8000)
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   ┌─────────────────┐               ┌─────────────────┐
   │ TensorFlow Model│               │ Prediction Logs │
   │ (Pipeline PKL)  │               │      (CSV)      │
   └─────────────────┘               └─────────────────┘
```

*Catatan: Apabila FastAPI Server offline, Streamlit memiliki sistem deteksi otomatis untuk melakukan fallback dan memproses prediksi secara lokal (client-side).*

---

## 🚀 Cara Menjalankan Aplikasi di Lokal

### 1. Prasyarat (Prerequisites)
Pastikan Anda menggunakan Python versi 3.10 atau 3.11.

### 2. Instalasi Dependensi
Aktifkan virtual environment Anda dan pasang pustaka yang diperlukan:
```bash
# Aktifkan virtual environment (Windows)
venv\Scripts\activate

# Instal dependensi dari file requirements.txt
pip install -r app/requirements.txt
```

### 3. Menjalankan Semua Layanan
Jalankan skrip starter otomatis untuk menyalakan FastAPI dan Streamlit secara bersamaan:
```bash
python start.py
```

Setelah berjalan, Anda dapat mengakses:
- **Aplikasi Dashboard (Streamlit):** [http://localhost:8501](http://localhost:8501)
- **REST API (FastAPI Docs):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Log Inferensi:** [logs/prediction_logs.csv](logs/prediction_logs.csv)

---

## 📁 Struktur Repositori

```
.
├── app/
│   ├── api.py                 # Backend API (FastAPI)
│   ├── app.py                 # Frontend UI (Streamlit)
│   └── requirements.txt       # Daftar dependensi deployment
├── data/
│   └── Delivery_Logistics.csv # Dataset operasional logistik
├── logs/
│   └── prediction_logs.csv    # Log inferensi dari API (MLOps Monitoring)
├── models/
│   └── delivery_delay_model.pkl # Pipeline model TensorFlow yang disimpan
├── notebooks/
│   └── delivery_notebook.ipynb # Jupyter Notebook eksplorasi & pelatihan model
├── start.py                   # Script starter multi-proses (FastAPI + Streamlit)
└── README.md                  # Dokumentasi proyek
```
