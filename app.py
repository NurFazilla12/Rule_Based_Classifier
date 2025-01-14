import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Fungsi untuk klasifikasi menggunakan Rule-Based Classifier
def rule_based_classifier(row):
    if row['Age'] < 30 and row['BP'] == 'LOW':
        return 'DrugX'
    elif row['Age'] >= 30 and row['Cholesterol'] == 'HIGH':
        return 'DrugY'
    elif row['Sex'] == 'M' and row['Na_to_K'] > 20:
        return 'DrugA'
    else:
        return 'DrugC'

# Membaca dataset dari CSV
data = pd.read_csv('Classification.csv')

# Menambahkan kolom 'Drug' menggunakan Rule-Based Classifier
data['Drug'] = data.apply(rule_based_classifier, axis=1)

# CSS untuk mengubah warna latar belakang dan sidebar
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f5;  /* Warna latar belakang aplikasi */
    }
    .sidebar .sidebar-content {
        background-color: #2E7D32;  /* Warna latar belakang sidebar */
        color: white;  /* Warna teks sidebar */
    }
    .sidebar .sidebar-content h1 {
        color: white;  /* Warna judul sidebar */
    }
    .sidebar .sidebar-content .stRadio {
        color: white;  /* Warna teks radio button */
    }
    h1 {
        color: #2c3e50;  /* Warna judul utama */
    }
    h2 {
        color: #2980b9;  /* Warna subjudul */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", [
    "Home 🏥", 
    "Klasifikasi 🩺", 
    "Analisis EDA 📊"
])

if page == "Home 🏥":
    # Judul halaman About
    st.title("ℹ️ Tentang Aplikasi")
    
    # Menambahkan ikon aplikasi
    st.image("images/medical.jpg", caption="health is happiness", use_container_width=True)
    
    # Deskripsi aplikasi
    st.markdown("""
    Aplikasi ini adalah **Aplikasi Klasifikasi Rule-Based** yang dirancang untuk membantu pengguna 
    dalam mengklasifikasikan data berdasarkan beberapa fitur kesehatan. 
    Dengan menggunakan metode klasifikasi berbasis aturan, aplikasi ini memberikan rekomendasi 
    tentang jenis obat yang sesuai berdasarkan input pengguna.
    """)

    # Tujuan aplikasi
    st.subheader("🎯 Tujuan Aplikasi")
    st.write("""
    - Memberikan klasifikasi obat berdasarkan data kesehatan.
    - Menyediakan analisis eksplorasi data untuk pemahaman yang lebih baik.
    - Memudahkan pengguna dalam memasukkan data dan mendapatkan hasil klasifikasi.
    """)

    # Cara penggunaan
    st.subheader("🛠️ Cara Penggunaan")
    st.write("""
    1. Pilih halaman **Klasifikasi** untuk memasukkan data kesehatan Anda.
    2. Masukkan informasi yang diperlukan dan klik tombol **Klasifikasikan**.
    3. Lihat hasil klasifikasi dan analisis data di halaman **Analisis EDA**.
    4. Kunjungi halaman ini untuk informasi lebih lanjut tentang aplikasi """)

elif page == "Klasifikasi 🩺":
    # Judul aplikasi
    st.title("🩺 Aplikasi Klasifikasi dengan Decision Tree")
    st.markdown("### Masukkan data untuk klasifikasi")

    # Input data dari pengguna
    age = st.number_input("👤 Masukkan usia (Age):", min_value=0, max_value=100, value=30)
    sex = st.selectbox("🚻 Pilih jenis kelamin (Sex):", options=['M', 'F'])
    bp = st.selectbox("💓 Pilih tekanan darah (BP):", options=['LOW', 'NORMAL', 'HIGH'])
    cholesterol = st.selectbox("🩸 Pilih kadar kolesterol (Cholesterol):", options=['NORMAL', 'HIGH'])
    na_to_k = st.number_input("⚖️ Masukkan rasio Na/K (Na_to_K):", min_value=0.0, max_value=50.0, value=10.0)

    # Buat DataFrame dari input
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'BP': [bp],
        'Cholesterol': [cholesterol],
        'Na_to_K': [na_to_k]
    })

    # Mengubah variabel kategorikal menjadi numerik
    input_data = pd.get_dummies(input_data, columns=['Sex', 'BP', 'Cholesterol'], drop_first=True)

    # Memisahkan fitur dan target dari dataset
    X = data.drop('Drug', axis=1)
    y = data['Drug']

    # Mengubah variabel kategorikal menjadi numerik
    X = pd.get_dummies(X, columns=['Sex', 'BP', 'Cholesterol'], drop_first=True)

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Melatih model Decision Tree
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Klasifikasi
    if st.button("Klasifikasikan"):
        # Menyesuaikan input_data dengan kolom yang digunakan saat pelatihan
        input_data = input_data.reindex(columns=X.columns, fill_value=0)
        prediction = model.predict(input_data)
        
        st.subheader("✅ Hasil Klasifikasi")
        st.write(f"Obat yang direkomendasikan: {prediction[0]}")

        # Menampilkan laporan klasifikasi
        y_pred = model.predict(X_test)
        st.subheader("📊 Laporan Klasifikasi")
        st.text(classification_report(y_test, y_pred))

    # Menampilkan data contoh
    st.markdown("### 📋 Contoh Data dan Klasifikasi")
    st.write(data.head())

    # Menampilkan preview dataset
    st.markdown("### 🔍 Preview Dataset")
    st.write(data)

    # Penjelasan Metode
    st.markdown("### 📖 Penjelasan Metode Klasifikasi")
    st.write("""
    Metode yang digunakan dalam aplikasi ini adalah **Decision Tree Classifier**. 
    Decision Tree adalah algoritma yang digunakan untuk klasifikasi dan regresi 
    dengan membagi dataset menjadi subset berdasarkan nilai fitur. 
    """)

elif page == "Analisis EDA 📊":
    # Judul halaman EDA
    st.title("📊 Analisis Eksplorasi Data (EDA)")
    
    # Statistik Deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.write(data.describe())

    # Visualisasi distribusi fitur
    st.subheader(" 📊 Visualisasi Distribusi Fitur")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(data['Age'], bins=10, kde=True, ax=ax[0])
    ax[0].set_title('Distribusi Usia')
    ax[0].set_xlabel('Usia')
    ax[0].set_ylabel('Frekuensi')

    sns.boxplot(x='Drug', y='Na_to_K', data=data, ax=ax[1])
    ax[1].set_title('Boxplot Rasio Na/K Berdasarkan Klasifikasi')
    ax[1].set_xlabel('Klasifikasi (Drug)')
    ax[1].set_ylabel('Rasio Na/K')

    st.pyplot(fig)

    # Visualisasi hubungan antar fitur
    st.subheader("🔍 Visualisasi Hubungan Antar Fitur")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Age', y='Na_to_K', hue='Drug', palette='viridis')
    plt.title('Scatter Plot Usia vs Rasio Na/K')
    plt.xlabel('Usia')
    plt.ylabel('Rasio Na/K')
    st.pyplot(plt)

    # Visualisasi distribusi klasifikasi
    st.subheader("📊 Distribusi Klasifikasi")
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x='Drug', palette='viridis')
    plt.title('Distribusi Klasifikasi dalam Dataset')
    plt.xlabel('Klasifikasi (Drug)')
    plt.ylabel('Jumlah')
    st.pyplot(plt)

    # Menampilkan data contoh
    st.markdown("### 📋 Contoh Data")
    st.write(data.head())

    # Penjelasan EDA
    st.markdown("### 📖 Penjelasan Analisis EDA")
    st.write("""
    Analisis Eksplorasi Data (EDA) bertujuan untuk memberikan pemahaman yang lebih baik tentang dataset. 
    Dengan menggunakan visualisasi dan statistik deskriptif, kita dapat mengidentifikasi pola, 
    tren, dan hubungan antar fitur yang ada dalam dataset.
    """)