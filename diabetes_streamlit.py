import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from streamlit_folium import folium_static

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Buat tampilan web menggunakan Streamlit
st.title('Aplikasi Prediksi Diabetes')

# Load dataset langsung di dalam kode
@st.cache_data
def load_data():
    return pd.read_csv('diabetes_skripsi.csv')

df = load_data()

# Buat sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ("Informasi Dataset", "Visualisasi", "Model LSTM", "Prediksi Diabetes"))

# Fungsi untuk menampilkan halaman awal
if page == "Informasi Dataset":
    st.header("Pengertian Diabetes")
    st.write(
        """
        Diabetes adalah penyakit kronis yang terjadi ketika tubuh tidak dapat memproduksi cukup insulin atau tidak dapat menggunakan insulin secara efektif.
        Insulin adalah hormon yang mengatur kadar glukosa dalam darah.
        """
    )
    st.header("Informasi Umum Dataset")
    st.write(
        """
        Dataset ini berisi informasi tentang pasien yang memiliki risiko diabetes.
        Fitur-fitur dalam dataset ini meliputi umur, tekanan darah, BMI, dan beberapa indikator kesehatan lainnya.
        """
    )

# Fungsi untuk menampilkan halaman dashboard visualisasi
elif page == "Visualisasi":
    st.header("Dashboard Visualisasi")

    # Load dataset
    df = pd.read_csv("diabetes_skripsi.csv")  # Ganti dengan path dataset Anda

    # 1. PEMETAAN MENGGUNAKAN FOLIUM
    st.subheader("Pemetaan Kasus Diabetes di Kota Bogor")

    # Create a map centered on Kota Bogor
    map_bogor = folium.Map(location=[-6.595038, 106.816635], zoom_start=12)

    # Create a MarkerCluster object
    marker_cluster = MarkerCluster().add_to(map_bogor)

    # Iterate through the DataFrame and add markers to the cluster
    for index, row in df.iterrows():
        try:
            latitude = float(row['latitude'])
            longitude = float(row['longitude'])
            kelurahan = row['kelurahan']
            umur = row['umur']
            jk = row['jk']
            diagnosis = row['diagnosis']

            # Customize marker color based on diagnosis
            if diagnosis == 1:
                color = 'red'
            else:
                color = 'blue'

            # Create a popup with information
            popup_text = f"<b>Kelurahan:</b> {kelurahan}<br><b>Umur:</b> {umur}<br><b>Jenis Kelamin:</b> {jk}<br><b>Diagnosis:</b> {diagnosis}"

            # Add a marker to the cluster with the popup and color
            folium.Marker(
                location=[latitude, longitude],
                popup=popup_text,
                icon=folium.Icon(color=color)
            ).add_to(marker_cluster)
        except (ValueError, KeyError) as e:
            print(f"Error processing row {index}: {e}")
            continue  # Skip this row if there's an error

    # Display the map
    folium_static(map_bogor)

    # 2. Distribusi Diagnosa Diabetes di Setiap Kelurahan
    st.subheader("Distribusi Diagnosa Diabetes di Setiap Kelurahan")
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='kelurahan', hue='diagnosis', palette='Set1')
    plt.xticks(rotation=90)
    plt.title('Distribusi Diagnosa Diabetes di Setiap Kelurahan')
    plt.xlabel('Kelurahan')
    plt.ylabel('Jumlah Kasus')
    st.pyplot(plt.gcf())

    # 3. Perbandingan Jumlah Diagnosa Diabetes dan Non-Diabetes (Pie Chart)
    st.subheader("Perbandingan Jumlah Diagnosa Diabetes dan Non-Diabetes")
    plt.figure(figsize=(6, 6))
    df['diagnosis'].value_counts().plot.pie(autopct='%1.1f%%', colors=['red', 'green'], startangle=90, explode=[0.1, 0])
    plt.title('Perbandingan Jumlah Diagnosa Diabetes dan Non-Diabetes')
    plt.ylabel('')
    st.pyplot(plt.gcf())

    # 4. Heatmap Korelasi
    st.subheader("Heatmap Korelasi Antar Variabel Numerik")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_columns) > 0:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Heatmap Korelasi Antar Variabel Numerik')
        st.pyplot(plt.gcf())

    # 5. Jumlah Pasien Diabetes Berdasarkan Jenis Kelamin
    st.subheader("Jumlah Pasien Diabetes Berdasarkan Jenis Kelamin")
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='jk', hue='diagnosis', palette='Set1')
    plt.title('Jumlah Pasien Diabetes Berdasarkan Jenis Kelamin')
    plt.xlabel('Jenis Kelamin')
    plt.ylabel('Jumlah Kasus')
    plt.legend(title='Diagnosis')
    st.pyplot(plt.gcf())

# Latih model LSTM
elif page == "Model LSTM":
    st.header('Latih Model LSTM')
    
    # Delete columns 'puskesmas', 'kelurahan', 'longitude', and 'latitude'
    df = df.drop(['puskesmas', 'kelurahan', 'longitude', 'latitude'], axis=1)
    
    # Define features (X) and target (y)
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']
    
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    # Simpan scaler di session state agar bisa digunakan di bagian lain aplikasi
    st.session_state['scaler'] = scaler
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_scaled, y)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_smote, 
                                                    y_smote, 
                                                    test_size=0.2, 
                                                    random_state=42)
    
    # Reshape data for LSTM
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    # Inisilisasi Hyperparameter
    neurons = 150
    epochs = 150
    batch_size = 128
    learning_rate = 0.001
    
    st.write(f"Jumlah Neuron: {neurons}")
    st.write(f"Jumlah Epoch: {epochs}")
    st.write(f"Batch Size: {batch_size}")
    st.write(f"Learning Rate: {learning_rate}")
    
    # Tambahkan tombol untuk melatih model
    if st.button('Latih Model'):
        # Membangun Model LSTM
        model = Sequential()
        model.add(LSTM(neurons, activation='relu', return_sequences=True, input_shape=(1, X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(neurons, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile Model dengan Optimizer
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
        
        # Latih model
        history = model.fit(X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test))
        
        # Simpan model ke dalam session state setelah dilatih
        st.session_state['model'] = model
        
        # Plot accuracy dan loss
        st.subheader('Grafik Akurasi dan Loss')
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].plot(history.history['accuracy'], label='Train Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()
        ax[0].set_title('Accuracy')

        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].legend()
        ax[1].set_title('Loss')

        st.pyplot(fig)
        
        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"Test Accuracy: {test_accuracy}")
        
        # Setelah model dilatih, kita lakukan prediksi pada test set
        y_pred_prob = model.predict(X_test)

        # Konversi prediksi probabilitas menjadi nilai kelas biner (0 atau 1)
        y_pred = (y_pred_prob > 0.5).astype("int32")

        # Hitung MAE, RMSE, dan MAPE berdasarkan probabilitas prediksi
        mae = mean_absolute_error(y_test, y_pred_prob)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_prob))

        # Print hasil evaluasi
        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")

# Fungsi untuk menampilkan halaman prediksi
elif page == "Prediksi Diabetes":
    st.header('Input Data Baru untuk Prediksi Diabetes')
    
    # Pastikan model dan scaler sudah dilatih dan disimpan di session state
    if 'model' not in st.session_state or 'scaler' not in st.session_state:
        st.warning("Model atau scaler belum dilatih. Silakan latih model terlebih dahulu di halaman 'Model LSTM'.")
    else:
        # Membuat dua kolom untuk form input
        col1, col2 = st.columns(2)

        # Input form untuk data baru di kolom 1
        with col1:
            umur = st.number_input("Umur:", min_value=0, max_value=120, value=0)
            jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            merokok = st.selectbox("Merokok", ["Ya", "Tidak"])
            aktivitas_fisik = st.selectbox("Aktivitas Fisik", ["Ya", "Tidak"])
            konsumsi_alkohol = st.selectbox("Konsumsi Alkohol", ["Ya", "Tidak"])

        # Input form untuk data baru di kolom 2
        with col2:
            tekanan_darah = st.number_input("Tekanan Darah:", min_value=0, value=0)
            bmi = st.number_input("BMI:", min_value=0.0, value=0.0)
            lingkar_perut = st.number_input("Lingkar Perut (cm)", min_value=0, max_value=200, value=0)
            pemeriksaan_gula = st.number_input("Hasil Pemeriksaan Gula (mg/dL)", min_value=0, max_value=400, value=0)

        # Konversi input ke format numerik
        jk = 0 if jk == "Laki-laki" else 1
        merokok = 1 if merokok == "Ya" else 0
        aktivitas_fisik = 1 if aktivitas_fisik == "Ya" else 0
        konsumsi_alkohol = 1 if konsumsi_alkohol == "Ya" else 0

        # Tombol untuk melakukan prediksi
        if st.button("Prediksi"):
            # Persiapkan data untuk prediksi
            new_data = pd.DataFrame({
                'umur': [umur],
                'jk': [jk],
                'merokok': [merokok],
                'aktivitas_fisik': [aktivitas_fisik],
                'konsumsi_alkohol': [konsumsi_alkohol],
                'tekanan_darah': [tekanan_darah],
                'bmi': [bmi],
                'lingkar_perut': [lingkar_perut],
                'pemeriksaan_gula': [pemeriksaan_gula]
            })

            # Gunakan scaler dari session state untuk transformasi data baru
            scaler = st.session_state['scaler']
            new_data_scaled = scaler.transform(new_data)
            
            # Reshape untuk cocok dengan input model (batch_size, timesteps, features)
            new_data_scaled = new_data_scaled.reshape((new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))

            # Prediksi menggunakan model
            model = st.session_state['model']
            new_prediction_prob = model.predict(new_data_scaled)
            new_prediction_class = (new_prediction_prob > 0.5).astype("int32")

            # Tampilkan hasil prediksi
            st.write(f"Probabilitas Diabetes: {new_prediction_prob[0][0]}")
            st.write(f"Prediksi Kelas: {'Diabetes' if new_prediction_class[0][0] == 1 else ' Tidak Diabetes'}")
