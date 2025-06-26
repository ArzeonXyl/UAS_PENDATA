import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide", page_title="Aplikasi Analisis Penyakit Hati")

st.title("Aplikasi Analisis Penyakit Hati (ILPD)")
st.markdown("Aplikasi ini menganalisis dataset pasien hati India (ILPD) menggunakan beberapa model Machine Learning.")

# --- Bagian Mendapatkan Data ---
st.header("1. Mendapatkan dan Memahami Data")

with st.expander("Lihat Detail Data Fetching"):
    st.write("Mengambil dataset ILPD (Indian Liver Patient Dataset) dari UCI ML Repository.")
    ilpd_indian_liver_patient_dataset = fetch_ucirepo(id=225)

    X = ilpd_indian_liver_patient_dataset.data.features
    y = ilpd_indian_liver_patient_dataset.data.targets

    st.subheader("Metadata Dataset:")
    st.write(ilpd_indian_liver_patient_dataset.metadata)

    st.subheader("Informasi Variabel:")
    st.write(ilpd_indian_liver_patient_dataset.variables)

st.subheader("Beberapa Baris Pertama DataFrame X (Fitur):")
st.dataframe(X.head())

st.subheader("Beberapa Baris Pertama DataFrame y (Target - 'Selector'):")
st.dataframe(y.head())

st.subheader("Kolom pada DataFrame X:")
st.write(X.columns.tolist()) # Menggunakan tolist() agar lebih rapi di Streamlit

st.subheader("Kolom pada DataFrame y:")
st.write(y.columns.tolist())

st.subheader("Deskripsi Tipe Data di Setiap Kolom:")
st.dataframe(ilpd_indian_liver_patient_dataset.variables[['name', 'role', 'type', 'description']])

st.subheader("Statistik Deskriptif DataFrame X:")
st.dataframe(X.describe())

st.subheader("Distribusi Kelas pada Kolom 'Selector' (y):")
st.write(y['Selector'].value_counts())

# Plot Countplot untuk 'Selector'
st.subheader("Visualisasi Distribusi Kelas 'Selector'")
fig_selector_count = plt.figure(figsize=(6, 4))
sns.countplot(data=y, x='Selector')
plt.title('Count of Selector Categories')
plt.xlabel('Selector')
plt.ylabel('Count')
st.pyplot(fig_selector_count)

# Plot Histogram untuk Fitur X
st.subheader("Distribusi Fitur Numerik (Histograms)")
fig_hist_x = plt.figure(figsize=(12, 10))
X.hist(ax=fig_hist_x.gca()) # Meneruskan axes object ke hist
plt.tight_layout()
st.pyplot(fig_hist_x)

# --- Bagian Preprocessing ---
st.header("2. Preprocessing Data")

# Menggabungkan X dan y untuk preprocessing awal
df = X.copy()
df['Selector'] = y['Selector']

st.subheader("Missing Values")
st.write("Jumlah missing values sebelum imputasi:")
st.write(df.isnull().sum())

# Imputasi Missing Values
df['A/G Ratio'].fillna(df['A/G Ratio'].median(), inplace=True)
st.write("Jumlah missing values setelah imputasi (A/G Ratio diisi dengan median):")
st.write(df.isnull().sum())

st.subheader("Encoding Fitur Kategorikal (Gender)")
# Mengubah Gender menjadi numerik (Male -> 1, Female -> 0)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
st.write("Dataframe setelah encoding Gender:")
st.dataframe(df.head())

# Memisahkan kembali fitur dan target setelah preprocessing awal
X_processed = df.drop('Selector', axis=1)
y_processed = df['Selector']

st.subheader("Penskalaan Fitur (StandardScaler)")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)
X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)
st.write("Dataframe fitur setelah penskalaan:")
st.dataframe(X_scaled.head())

# --- Deteksi Outlier dengan LOF ---
st.subheader("Deteksi dan Penghapusan Outlier (Local Outlier Factor - LOF)")

with st.expander("Pelajari Konsep LOF"):
    st.markdown("""
    LOF (Local Outlier Factor) adalah metode deteksi outlier yang bersifat **lokal**. Ia mengukur kepadatan (density) di sekitar sebuah titik dan membandingkannya dengan kepadatan rata-rata di sekitar tetangga-tetangganya. Sebuah titik dianggap outlier jika ia berada di area yang **jauh lebih tidak padat (less dense)** daripada area di mana tetangganya berada.

    **Konsep Kunci dan Rumus Matematis:**
    * **Parameter `k`:** Jumlah tetangga terdekat yang akan dipertimbangkan.
    * **1. Jarak ke-k (k-distance):** Jarak dari titik A ke tetangga terdekatnya yang ke-`k`. Ini mendefinisikan "radius" lingkungan lokal di sekitar titik A.
    * **2. Jarak Ketercapaian (Reachability Distance):** $\text{reachability-distance}_k(A, B) = \max \{ k\text{-distance}(B), d(A, B) \}$
        * Tujuannya untuk "meratakan" jarak, mencegah hasil yang tidak stabil jika titik-titik berada dalam satu klaster yang sangat padat.
    * **3. Kepadatan Ketercapaian Lokal (Local Reachability Density - lrd):** $\text{lrd}_k(A) = \frac{1}{\frac{\sum_{B \in N_k(A)} \text{reachability-distance}_k(A, B)}{|N_k(A)|}}$
        * Jika `lrd` tinggi, area padat. Jika `lrd` rendah, area renggang.
    * **4. Local Outlier Factor (LOF) - Rumus Akhir:** $\text{LOF}_k(A) = \frac{\frac{\sum_{B \in N_k(A)} \text{lrd}_k(B)}{|N_k(A)|}}{\text{lrd}_k(A)}$

    **Interpretasi Nilai LOF:**
    * **LOF ≈ 1:** Inlier (bukan outlier), kepadatan titik A mirip tetangga.
    * **LOF < 1:** Inlier, titik A berada di area sangat padat.
    * **LOF > 1:** Kandidat kuat outlier, titik A berada di area lebih renggang dari lingkungannya. Semakin besar nilainya, semakin tinggi kemungkinan outlier.

    **Kelebihan & Kekurangan LOF:**
    * **Kelebihan:** Efektif pada data dengan kepadatan bervariasi, tidak perlu asumsi distribusi, memberikan skor.
    * **Kekurangan:** Komputasi mahal pada dataset besar, sensitif terhadap parameter `k`, kurang efektif pada data berdimensi tinggi.
    """)

k_lof = st.slider("Pilih nilai k (n_neighbors) untuk LOF:", min_value=1, max_value=50, value=3)
contamination_lof = st.slider("Pilih Contamination (proporsi outlier yang diharapkan):", min_value=0.01, max_value=0.5, value=0.2, step=0.01)
threshold_lof = st.slider("Pilih Threshold untuk LOF Score:", min_value=1.0, max_value=3.0, value=1.8, step=0.1)


numeric_cols_for_lof = X_scaled.select_dtypes(include=np.number).columns.tolist()
if 'Gender' in numeric_cols_for_lof:
    numeric_cols_for_lof.remove('Gender') # Asumsi Gender sudah di-encode tapi tidak ingin dihitung dalam LOF jika dianggap kategorikal

X_scaled_numeric_for_lof = X_scaled[numeric_cols_for_lof]

lof = LocalOutlierFactor(n_neighbors=k_lof, contamination=contamination_lof)
lof.fit(X_scaled_numeric_for_lof)
outlier_scores = -lof.negative_outlier_factor_

# Menentukan status outlier berdasarkan threshold
is_outlier_status = np.where(outlier_scores > threshold_lof, -1, 1)

X_scaled['lof_score'] = outlier_scores
X_scaled['is_outlier'] = is_outlier_status

st.write("Samples dengan skor LOF tertinggi (paling mungkin outlier):")
st.dataframe(X_scaled.sort_values(by='lof_score', ascending=False).head())

outlier_count = X_scaled[X_scaled['is_outlier'] == -1].shape[0]
inlier_count = X_scaled[X_scaled['is_outlier'] == 1].shape[0]

st.write(f"\nJumlah data yang terdeteksi sebagai outlier (-1) berdasarkan threshold {threshold_lof}: {outlier_count}")
st.write(f"Jumlah data yang terdeteksi sebagai inlier (1) berdasarkan threshold {threshold_lof}: {inlier_count}")

# Filter data untuk menghapus outlier
X_cleaned_scaled = X_scaled[X_scaled['is_outlier'] == 1].drop(['lof_score', 'is_outlier'], axis=1)
y_cleaned_encoded = y_processed.loc[X_cleaned_scaled.index]

st.write("Jumlah data setelah menghapus outlier:")
st.write(f"Fitur (X_cleaned_scaled): {len(X_cleaned_scaled)}")
st.write(f"Target (y_cleaned_encoded): {len(y_cleaned_encoded)}")

st.write("Data fitur setelah menghapus outlier dan penskalaan (beberapa baris pertama):")
st.dataframe(X_cleaned_scaled.head())

st.write("Data target setelah menghapus outlier (beberapa baris pertama):")
st.dataframe(y_cleaned_encoded.head())

# --- Pembagian Data Latih/Uji dan Transformasi Distribusi ---
st.subheader("Pembagian Data Latih/Uji dan Transformasi Distribusi (PowerTransformer)")

X_data = X_cleaned_scaled.copy()
y_data = y_cleaned_encoded.copy()

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

st.write("Ukuran data setelah dibagi (sebelum transformasi):")
st.write(f"X_train shape: {X_train.shape}")
st.write(f"X_test shape: {X_test.shape}")
st.write(f"y_train shape: {y_train.shape}")
st.write(f"y_test shape: {y_test.shape}")

numeric_cols_for_transform = X_train.select_dtypes(include=np.number).columns.tolist()
if 'Gender' in numeric_cols_for_transform:
    numeric_cols_for_transform.remove('Gender')

pt = PowerTransformer(method='yeo-johnson')

X_train_transformed = X_train.copy()
X_train_transformed[numeric_cols_for_transform] = pt.fit_transform(X_train[numeric_cols_for_transform])

X_test_transformed = X_test.copy()
X_test_transformed[numeric_cols_for_transform] = pt.transform(X_test[numeric_cols_for_transform])

st.write("Data fitur latih setelah transformasi distribusi (beberapa baris pertama):")
st.dataframe(X_train_transformed.head())

st.write("Data fitur uji setelah transformasi distribusi (beberapa baris pertama):")
st.dataframe(X_test_transformed.head())

# --- Bagian Model Machine Learning ---
st.header("3. Pemodelan Machine Learning")

# --- Naive Bayes ---
st.subheader("3.1. Naive Bayes")
with st.expander("Pelajari Konsep Naive Bayes"):
    st.markdown("""
    Naïve Bayes adalah algoritma klasifikasi yang didasarkan pada Teorema Bayes dengan asumsi "naif" mengenai independensi antar fitur.
    **Teorema Bayes:** $P(Y|X) = \frac{P(X|Y) * P(Y)}{P(X)}$
    **Asumsi "Naif":** $P(X|Y) = P(x_1|Y) * P(x_2|Y) * ... * P(x_n|Y)$
    **Proses Klasifikasi:** Prediksi kelas dengan probabilitas posterior tertinggi.
    **Varian:** Gaussian (untuk fitur kontinu), Multinomial (untuk fitur diskrit/count), Bernoulli (untuk fitur biner).
    **Kelebihan:** Cepat, efisien, tidak butuh banyak data training, baik untuk data berdimensi tinggi.
    **Kekurangan:** Asumsi independensi yang kuat, masalah *zero probability*, estimasi probabilitas kurang akurat.
    """)

X_train_data_nb = X_train_transformed.copy()
y_train_data_nb = y_train.copy()
X_test_data_nb = X_test_transformed.copy()
y_test_data_nb = y_test.copy()

nb_model = GaussianNB()
nb_model.fit(X_train_data_nb, y_train_data_nb)
likelihoods_test_nb = nb_model.predict_proba(X_test_data_nb)
final_prediction_test_nb = nb_model.predict(X_test_data_nb)

likelihood_df_test_nb = pd.DataFrame(likelihoods_test_nb, columns=[f'likelihood_class_{c}' for c in nb_model.classes_])
likelihood_df_test_nb.index = X_test_data_nb.index

data_test_results_nb = pd.concat([X_test_data_nb, likelihood_df_test_nb, y_test_data_nb], axis=1)
data_test_results_nb.rename(columns={'Selector': 'Actual_Selector'}, inplace=True)
data_test_results_nb["final_prediction"] = final_prediction_test_nb

st.write("Hasil Prediksi Naïve Bayes pada Data Uji (beberapa baris pertama):")
st.dataframe(data_test_results_nb.head())

accuracy_nb = accuracy_score(y_test_data_nb, final_prediction_test_nb)
st.write(f"Akurasi Model Naïve Bayes: {accuracy_nb * 100:.2f}%")

report_nb = classification_report(y_test_data_nb, final_prediction_test_nb, output_dict=True)
st.write("Laporan Klasifikasi Naïve Bayes:")
st.dataframe(pd.DataFrame(report_nb).transpose())

cm_nb = confusion_matrix(y_test_data_nb, final_prediction_test_nb)
fig_cm_nb = plt.figure(figsize=(6, 4))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix Naïve Bayes')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
st.pyplot(fig_cm_nb)


# --- Decision Tree ---
st.subheader("3.2. Decision Tree")
with st.expander("Pelajari Konsep Decision Tree"):
    st.markdown("""
    Decision Tree adalah algoritma klasifikasi/regresi supervised yang memecah dataset secara rekursif menjadi subset-subset.
    **Struktur Pohon:** Root Node, Internal Node (pengujian fitur), Branches (hasil pengujian), Leaf Node (label kelas/nilai prediksi).
    **Kriteria Pemecahan:** Gini Impurity atau Entropy (memilih fitur yang menghasilkan penurunan ketidakmurnian terbesar).
    **Kondisi Berhenti:** Node murni, tidak ada fitur tersisa, kedalaman maks tercapai, sampel min tercapai.
    **Kelebihan:** Mudah diinterpretasikan, tidak butuh penskalaan, tangani numerik/kategorikal, tangkap hubungan non-linier.
    **Kekurangan:** Cenderung *overfitting*, sensitif terhadap variasi data kecil, bias terhadap fitur dengan banyak kategori, tidak selalu optimal global.
    """)

X_train_data_dt = X_train_transformed.copy()
y_train_data_dt = y_train.copy()
X_test_data_dt = X_test_transformed.copy()
y_test_data_dt = y_test.copy()

model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train_data_dt, y_train_data_dt)
y_pred_dt = model_dt.predict(X_test_data_dt)

accuracy_dt = accuracy_score(y_test_data_dt, y_pred_dt)
st.write(f"Akurasi Model Decision Tree pada Data Uji: {accuracy_dt:.2%}")

report_dt = classification_report(y_test_data_dt, y_pred_dt, output_dict=True)
st.write("Laporan Klasifikasi Decision Tree:")
st.dataframe(pd.DataFrame(report_dt).transpose())

cm_dt = confusion_matrix(y_test_data_dt, y_pred_dt)
fig_cm_dt = plt.figure(figsize=(6, 4))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix Decision Tree')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
st.pyplot(fig_cm_dt)


# --- K-Nearest Neighbors (KNN) ---
st.subheader("3.3. K-Nearest Neighbors (KNN)")
with st.expander("Pelajari Konsep KNN"):
    st.markdown("""
    KNN adalah algoritma supervised machine learning "malas" untuk klasifikasi dan regresi.
    **Prinsip Dasar:** Klasifikasi berdasarkan mayoritas kelas dari *k* tetangga terdekat di dataset pelatihan.
    **Parameter k:** Jumlah tetangga terdekat yang dipertimbangkan.
    **Pengukuran Jarak:** Jarak Euclidean ($d(p, q) = \sqrt{\sum_{i=1}^n (q_i - p_i)^2}$) paling umum.
    **Proses Prediksi:** Hitung jarak ke semua titik pelatihan, pilih *k* terdekat, tentukan kelas mayoritas (klasifikasi) atau rata-rata nilai target (regresi).
    **Kelebihan:** Sederhana, tidak ada asumsi distribusi, fleksibel, efektif untuk dataset kecil-menengah.
    **Kekurangan:** Komputasi mahal saat prediksi, sensitif terhadap skala fitur (perlu penskalaan!), sensitif terhadap outlier, pemilihan nilai *k* menantang, tidak baik untuk data dimensi tinggi.
    """)

X_train_data_knn = X_train_transformed.copy()
y_train_data_knn = y_train.copy()
X_test_data_knn = X_test_transformed.copy()
y_test_data_knn = y_test.copy()

model_knn = KNeighborsClassifier()
model_knn.fit(X_train_data_knn, y_train_data_knn)
y_pred_knn = model_knn.predict(X_test_data_knn)

accuracy_knn = accuracy_score(y_test_data_knn, y_pred_knn)
st.write(f"Akurasi Model K-Nearest Neighbors pada Data Uji: {accuracy_knn:.2%}")

report_knn = classification_report(y_test_data_knn, y_pred_knn, output_dict=True)
st.write("Laporan Klasifikasi KNN:")
st.dataframe(pd.DataFrame(report_knn).transpose())

cm_knn = confusion_matrix(y_test_data_knn, y_pred_knn)
fig_cm_knn = plt.figure(figsize=(6, 4))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix KNN')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
st.pyplot(fig_cm_knn)

# --- Bagian Evaluasi ---
st.header("4. Evaluasi Model")

st.subheader("Ringkasan Akurasi Model pada Data Uji")
# Mendapatkan nilai precision, recall, f1-score untuk kelas 2 dari laporan
# Ini diasumsikan report_nb, report_dt, report_knn adalah dictionary dari classification_report(output_dict=True)

precision_nb_class_2 = report_nb['2']['precision'] if '2' in report_nb else 'N/A'
recall_nb_class_2 = report_nb['2']['recall'] if '2' in report_nb else 'N/A'
f1_nb_class_2 = report_nb['2']['f1-score'] if '2' in report_nb else 'N/A'

precision_dt_class_2 = report_dt['2']['precision'] if '2' in report_dt else 'N/A'
recall_dt_class_2 = report_dt['2']['recall'] if '2' in report_dt else 'N/A'
f1_dt_class_2 = report_dt['2']['f1-score'] if '2' in report_dt else 'N/A'

precision_knn_class_2 = report_knn['2']['precision'] if '2' in report_knn else 'N/A'
recall_knn_class_2 = report_knn['2']['recall'] if '2' in report_knn else 'N/A'
f1_knn_class_2 = report_knn['2']['f1-score'] if '2' in report_knn else 'N/A'


evaluation_data = {
    'Model': ['Gaussian Naive Bayes', 'Decision Tree', 'K-Nearest Neighbors (KNN)'],
    'Akurasi': [f"{accuracy_nb * 100:.2f}%", f"{accuracy_dt * 100:.2f}%", f"{accuracy_knn * 100:.2f}%"],
    'Precision (Kelas 2)': [f"{precision_nb_class_2 * 100:.2f}%", f"{precision_dt_class_2 * 100:.2f}%", f"{precision_knn_class_2 * 100:.2f}%"],
    'Recall (Kelas 2)': [f"{recall_nb_class_2 * 100:.2f}%", f"{recall_dt_class_2 * 100:.2f}%", f"{recall_knn_class_2 * 100:.2f}%"],
    'F1-Score (Kelas 2)': [f"{f1_nb_class_2 * 100:.2f}%", f"{f1_dt_class_2 * 100:.2f}%", f"{f1_knn_class_2 * 100:.2f}%"]
}
st.dataframe(pd.DataFrame(evaluation_data))

st.markdown("""
*Note: Angka akurasi, Precision, Recall, dan F1-Score diambil dari output sel-sel sebelumnya. Perbedaan kecil pada angka akurasi antara eksekusi mungkin terjadi.*
""")

# --- Bagian Kesimpulan dan Rekomendasi ---
st.header("5. Kesimpulan dan Langkah Selanjutnya")

st.markdown("""
Berdasarkan analisis dan pemodelan yang telah dilakukan pada **ILPD (Indian Liver Patient Dataset)**, kita telah melalui serangkaian langkah pra-pemrosesan data yang meliputi:

1.  Penanganan missing values pada kolom 'A/G Ratio'.
2.  Encoding fitur kategorikal 'Gender'.
3.  Penskalaan fitur menggunakan StandardScaler.
4.  Deteksi dan penghapusan outlier menggunakan Local Outlier Factor (LOF) dengan parameter spesifik yang dapat diatur pengguna.
5.  Transformasi distribusi pada fitur-fitur numerik menggunakan PowerTransformer (Yeo-Johnson) setelah pembagian data.
6.  Pembagian data menjadi set pelatihan dan pengujian dengan stratifikasi.

Setelah pra-pemrosesan ini, kita melatih dan mengevaluasi tiga model klasifikasi: **Gaussian Naive Bayes**, **Decision Tree**, dan **K-Nearest Neighbors (KNN)** pada data uji.
""")

st.subheader("Analisis Performa Model:")
st.markdown("""
* Secara akurasi keseluruhan pada data uji, **Model K-Nearest Neighbors (KNN)** menunjukkan performa terbaik, diikuti oleh Gaussian Naive Bayes dan Decision Tree.
* Meskipun Gaussian Naive Bayes memiliki precision yang sangat tinggi untuk Kelas 2 (menunjukkan bahwa ketika ia memprediksi positif, kemungkinannya benar sangat tinggi), Recall-nya sangat rendah, artinya model ini melewatkan sebagian besar kasus positif yang sebenarnya. Ini kemungkinan besar dipengaruhi oleh ketidakseimbangan kelas dan asumsi model.
* Model KNN menunjukkan keseimbangan yang lebih baik antara Precision dan Recall untuk Kelas 2 dibandingkan Naive Bayes dan Decision Tree, yang tercermin dalam F1-Score yang lebih tinggi untuk kelas minoritas.
* Decision Tree menunjukkan performa yang paling rendah di antara ketiganya dalam skenario ini.
""")

st.subheader("Kesimpulan Akhir:")
st.markdown("""
Berdasarkan evaluasi pada data uji, **K-Nearest Neighbors (KNN) adalah model yang paling menjanjikan** di antara ketiganya untuk tugas klasifikasi ini, memberikan akurasi keseluruhan terbaik dan keseimbangan metrik yang lebih baik untuk kelas minoritas. Pra-pemrosesan data yang komprehensif, termasuk penanganan outlier dan transformasi distribusi, kemungkinan berkontribusi pada performa ini.
""")

st.subheader("Langkah Selanjutnya yang Direkomendasikan:")
st.markdown("""
1.  **Tuning Hyperparameter KNN:** Fokus pada model KNN dan lakukan pencarian hyperparameter terbaik (terutama nilai `n_neighbors` atau `k`) menggunakan cross-validation untuk mengoptimalkan performanya lebih lanjut.
2.  **Eksplorasi Metrik Lain:** Selain akurasi, fokus pada metrik spesifik seperti AUC-ROC jika relevan untuk masalah ini.
3.  **Coba Model Lanjutan:** Pertimbangkan algoritma klasifikasi yang lebih canggih seperti Support Vector Machines (SVM), Random Forest, atau Gradient Boosting (XGBoost, LightGBM) yang seringkali berkinerja baik pada dataset seperti ini.
4.  **Penanganan Imbalance:** Jika Recall kelas minoritas masih menjadi perhatian utama, eksplorasi teknik penanganan imbalance (SMOTE dengan parameter berbeda, teknik lain, atau weighted loss functions pada model yang mendukung).
5.  **Feature Engineering/Selection:** Analisis lebih dalam terhadap fitur atau pembuatan fitur baru mungkin dapat meningkatkan kemampuan model.
""")