# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Cybersecurity Analysis",
    page_icon="🔒",
    layout="wide"
)

# Sidebar Styling
st.sidebar.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# === STEP 1: Load and Preprocess Data ===
st.sidebar.title("🔍 Navigasi")
st.sidebar.info("Pilih halaman untuk menjelajahi data atau menjalankan pemodelan.")

# File path untuk dataset
file_path = "C:/Users/ASUSTeK/DM/data/cybersecurity_attacks.csv"
data = pd.read_csv(file_path)

# Preprocessing
columns_to_drop = ['Alerts/Warnings', 'IDS/IPS Alerts', 'Malware Indicators', 
                   'Firewall Logs', 'Proxy Information']
data_cleaned = data.drop(columns=columns_to_drop, errors='ignore')

# Ekstraksi fitur waktu
if 'Timestamp' in data_cleaned.columns:
    data_cleaned['Timestamp'] = pd.to_datetime(data_cleaned['Timestamp'], errors='coerce')
    data_cleaned['Hour'] = data_cleaned['Timestamp'].dt.hour
    data_cleaned['Day'] = data_cleaned['Timestamp'].dt.day
    data_cleaned['Month'] = data_cleaned['Timestamp'].dt.month
    data_cleaned.drop(columns=['Timestamp'], inplace=True)

# Encode kolom non-numerik
non_numeric_columns = data_cleaned.select_dtypes(include=['object']).columns
label_encoders = {}
for column in non_numeric_columns:
    le = LabelEncoder()
    data_cleaned[column] = le.fit_transform(data_cleaned[column])
    label_encoders[column] = le

# === STEP 2: Sidebar Navigation ===
menu = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["🏠 Beranda", "🛠 Pengolahan Data", "📊 Exploratory Data Analysis (EDA)", "🤖 Pemodelan"]
)

# === STEP 3: Pages ===

# Beranda
if menu == "🏠 Beranda":
    st.title("🔒 Cybersecurity Analysis")
    st.markdown(
        """
        **Selamat Datang!**  
        Aplikasi ini memperlihatkan Anda untuk:
        - Membersihkan dan memproses data serangan siber.
        - Mengeksplorasi data melalui analisis statistik dan visualisasi.
        - Membangun model klasifikasi untuk mendeteksi serangan.  
        
        **Langkah-langkah Penggunaan:**
        1. Pilih halaman di navigasi.
        2. Ikuti proses di masing-masing bagian.
        """
    )
    st.image("https://th.bing.com/th/id/OIP.u9rmMa7F-vTpCRI4184ldAHaDQ?rs=1&pid=ImgDetMain", caption="Cybersecurity in Action", use_column_width=True)

# Pengolahan Data
elif menu == "🛠 Pengolahan Data":
    st.title("🛠 Pengolahan Data")
    st.markdown("**Hasil pembersihan data dan ringkasan nilai yang hilang.**")
    
    st.subheader("📋 Preview Data Bersih")
    st.dataframe(data_cleaned.head())

    st.subheader("📉 Missing Values")
    missing_data = data_cleaned.isnull().sum()
    st.write(missing_data)

# Exploratory Data Analysis (EDA)
elif menu == "📊 Exploratory Data Analysis (EDA)":
    st.title("📊 Exploratory Data Analysis (EDA)")

    # Statistik deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.write(data_cleaned.describe())
    
    # Heatmap korelasi
    st.subheader("🌐 Heatmap Korelasi")
    corr_matrix = data_cleaned.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Pilihan untuk distribusi data
    st.subheader("📊 Distribusi Data Numerik")
    numerical_columns = ['Source Port', 'Destination Port', 'Packet Length', 
                         'Anomaly Scores', 'Hour', 'Day', 'Month']
    selected_column = st.selectbox("Pilih Kolom untuk Distribusi:", numerical_columns)
    if selected_column in data_cleaned.columns:
        fig, ax = plt.subplots()
        sns.histplot(data_cleaned[selected_column], kde=True, bins=30, color='blue', ax=ax)
        st.pyplot(fig)

# Pemodelan
elif menu == "🤖 Pemodelan":
    st.title("🤖 Pemodelan")
    st.markdown("**Membangun model untuk klasifikasi serangan siber.**")
    
    # Pemilihan target dan fitur
    target_column = 'Attack Type'
    features = ['Source Port', 'Destination Port', 'Packet Length', 
                'Anomaly Scores', 'Hour', 'Day', 'Month']
    
    # Encode target jika perlu
    if target_column in data_cleaned.columns:
        le_target = LabelEncoder()
        data_cleaned[target_column] = le_target.fit_transform(data_cleaned[target_column])

    # Drop data hilang
    data_cleaned.dropna(subset=features + [target_column], inplace=True)
    X = data_cleaned[features]
    y = data_cleaned[target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random Forest Classifier
    st.subheader("⚙️ Training Random Forest Model")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi model
    st.subheader("🔍 Evaluasi Model")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Akurasi Model", f"{accuracy_score(y_test, y_pred):.2%}")
    with col2:
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
