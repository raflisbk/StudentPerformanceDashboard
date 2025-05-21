# Student Performance Dashboard

![Dashboard Screenshot](https://cdn-icons-png.flaticon.com/512/3413/3413535.png)

## Deskripsi Proyek

Dashboard Student Performance adalah aplikasi web interaktif yang dibangun dengan Streamlit untuk menganalisis dan memvisualisasikan data performa mahasiswa. Aplikasi ini menggunakan teknik machine learning untuk mengidentifikasi pola dalam data mahasiswa dan memprediksi risiko putus kuliah (dropout).

Dashboard ini menyediakan berbagai fitur termasuk:

- **Eksplorasi Data**: Visualisasi interaktif untuk memahami karakteristik demografis, performa akademik, dan faktor sosial-ekonomi mahasiswa
- **Analisis Clustering**: Pengelompokan mahasiswa berdasarkan karakteristik mereka untuk identifikasi pola risiko
- **Prediksi Risiko Dropout**: Model prediktif untuk mengidentifikasi mahasiswa yang berisiko putus kuliah berdasarkan profil dan performa akademik mereka

## Sumber Data

Dataset yang digunakan dalam proyek ini berasal dari institusi pendidikan tinggi dan berisi informasi tentang mahasiswa yang terdaftar di berbagai program sarjana, termasuk informasi demografis, jalur akademik, faktor sosial-ekonomi, dan indikator performa akademik.

**Sumber**: Realinho, Valentim, Vieira Martins, Mónica, Machado, Jorge, and Baptista, Luís. (2021). Predict students' dropout and academic success. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

## Fitur Utama

1. **Beranda**: Gambaran umum tentang dashboard dengan metrik utama dan ringkasan distribusi risiko
2. **Eksplorasi Data**: Visualisasi dan analisis mendalam tentang karakteristik mahasiswa dengan tab terorganisir:
   - Demografi Mahasiswa
   - Performa Akademik
   - Faktor Sosial-Ekonomi
3. **Analisis Clustering**: Pengelompokan menggunakan algoritma MeanShift untuk mengidentifikasi pola alami dalam data mahasiswa
4. **Prediksi Risiko Dropout**: Form interaktif untuk memprediksi tingkat risiko dropout berdasarkan informasi mahasiswa
5. **Tentang**: Informasi tentang proyek, metodologi, dan kontak

## Metodologi

Proyek ini menggunakan pendekatan data science yang komprehensif:

1. **Preprocessing Data**: Pembersihan, normalisasi, dan pembuatan fitur untuk menyiapkan data
2. **Analisis Data Eksploratif**: Visualisasi dan pemahaman hubungan antar variabel
3. **Analisis Clustering**: Penggunaan algoritma MeanShift untuk mengidentifikasi pengelompokan alami mahasiswa
4. **Model Klasifikasi**: Pembuatan model prediktif untuk mengidentifikasi mahasiswa berisiko dropout
5. **Penilaian Risiko**: Penentuan tingkat risiko (Rendah, Menengah, Tinggi) berdasarkan karakteristik kluster dan prediksi model

## Teknologi yang Digunakan

- **Python**: Bahasa pemrograman utama
- **Streamlit**: Framework untuk membuat aplikasi web
- **Pandas & NumPy**: Manipulasi dan analisis data
- **Plotly**: Visualisasi data interaktif
- **Scikit-learn**: Implementasi model machine learning

## Struktur Proyek

student-performance-dashboard/
├── app.py                    # File utama aplikasi Streamlit
├── README.md                 # Dokumentasi proyek
├── assets/                   # Aset untuk UI
│   └── css/
│       └── style.css         # CSS kustom
├── data/                     # File data
│   ├── data.csv
│   ├── clustering_data.csv
│   └── data_with_risk_labels.csv
├── models/                   # Model ML yang disimpan
│   ├── meanshift_model.pkl
│   └── svm_risk_category_model.pkl
├── pages_content/            # Konten halaman untuk aplikasi
│   ├── home.py
│   ├── data_exploration.py
│   ├── clustering.py
│   ├── classification.py
│   └── about.py
└── utils/                    # Modul utilitas
├── load_data.py
├── preprocessing.py
├── clustering.py
└── classification.py


## Cara Menjalankan

1. Pastikan Python 3.8+ sudah terinstal di sistem Anda
2. Kloning repositori ini
3. Instal dependensi yang dibutuhkan: pip install -r requirements.txt
4. Jalankan aplikasi Streamlit: streamlit run app.py
5. Buka browser dan akses `http://localhost:8501`

## Pembagian Risiko

Dashboard mengklasifikasikan mahasiswa ke dalam tiga kategori risiko dropout:

- **Risiko Tinggi**: Mahasiswa dengan kemungkinan dropout yang tinggi, biasanya dengan rasio kelulusan rendah, nilai masuk rendah, dan masalah pembayaran uang kuliah
- **Risiko Menengah**: Mahasiswa dengan indikator performa campuran, memerlukan pemantauan periodik
- **Risiko Rendah**: Mahasiswa dengan performa akademik kuat dan cenderung menyelesaikan studi dengan sukses

## Pengembangan Lebih Lanjut

Beberapa ide untuk pengembangan dashboard di masa depan:

- Integrasi dengan sistem informasi mahasiswa untuk data real-time
- Fitur peringatan otomatis untuk mahasiswa berisiko tinggi
- Analisis longitudinal untuk melacak performa mahasiswa dari waktu ke waktu
- Rekomendasi intervensi yang lebih dipersonalisasi berdasarkan karakteristik spesifik mahasiswa

## Kontak

**Nama**: Mohamad Rafli Agung Subekti
**Email**: raflisbk@gmail.com

## Lisensi

Proyek ini bersifat open-source dan tersedia di bawah Lisensi MIT. Silakan merujuk ke file LICENSE untuk informasi lebih lanjut.
