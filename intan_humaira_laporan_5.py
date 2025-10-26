# -*- coding: utf-8 -*-
"""HoloVision Dashboard - Maroon Theme (Fixed but Original Content Kept)"""

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="HoloFruits Vision Dashboard", layout="wide")

# ==========================
# CSS STYLING (SOFT MAROON AURORA THEME)
# ==========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8dede, #f3cfcf, #e0b1b1, #b87b7b);
    background-size: 400% 400%;
    animation: gradientMove 25s ease infinite;
    position: relative;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Kartu kaca transparan */
.glass-card {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 4px 25px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(12px);
    margin-bottom: 25px;
}

/* Judul besar */
h1 {
    text-align: center;
    font-size: 2.3em;
    color: #541818;
    font-weight: 800;
    background: rgba(255,255,255,0.65);
    padding: 15px;
    border-radius: 15px;
    display: inline-block;
    box-shadow: 0 2px 15px rgba(0,0,0,0.05);
}

h3 {
    color: #4b1c1c;
}

.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 0.9em;
    color: #5e3a3a;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
logo_candidates = [
    ".devcontainer/usk_logo.png",
    ".devcontainer/logo_usk.png",
    "assets/usk_logo.png",
    "usk_logo.png"
]
logo_path = next((p for p in logo_candidates if os.path.exists(p)), None)

col1, col2 = st.columns([0.15, 0.85])
with col1:
    if logo_path:
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown("<div style='width:90px;height:90px;background:#a87b7b;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;'>USK</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<h1>HoloFruits Vision Dashboard</h1>", unsafe_allow_html=True)

# ==========================
# DESKRIPSI DATASET 📚 (TIDAK DIUBAH)
# ==========================
st.markdown("""
<div class="glass-card">
    <h3>🧺 Deskripsi Dataset: <em>Fruits Fresh and Rotten for Classification</em></h3>
    <p style='text-align: justify;'>
    Dataset ini berasal dari platform 
    <a href='https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification' target='_blank'>Kaggle</a>. 
    Dataset ini berisi kumpulan gambar buah-buahan dalam dua kondisi, yaitu fresh (segar) dan rotten (busuk), 
    mencakup tiga jenis buah: apel, pisang, dan jeruk. 
    Setiap kombinasi menghasilkan enam kelas gambar sebagai berikut:
        <li>freshapples</li>
        <li>rottenapples</li>
        <li>freshbanana</li>
        <li>rottenbanana</li>
        <li>freshoranges</li>
        <li>rottenoranges</li>
    Dataset terbagi menjadi dua bagian utama:
    Train: 10.901 gambar dan
    Test: 2.698 gambar
    <br>
    Tujuan utama dataset ini adalah untuk melatih dan menguji model klasifikasi gambar 
    agar dapat mengenali kondisi buah berdasarkan penampilan visualnya. 
    Dataset ini banyak digunakan dalam penelitian bidang Computer Vision dan Deep Learning 
    menggunakan arsitektur Convolutional Neural Network (CNN).
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_path = "Model/Intan Humaira_Laporan 4.pt"
        keras_path = "Model/Intan Humaira_Laporan2.h5"

        yolo_model = YOLO(yolo_path) if os.path.exists(yolo_path) else None
        classifier = tf.keras.models.load_model(keras_path) if os.path.exists(keras_path) else None
        return yolo_model, classifier
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None

yolo_model, classifier = load_models()

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("🎛️ Mode Analisis")
mode = st.sidebar.selectbox("Pilih Fungsi:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.sidebar.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# KONTEN UTAMA
# ==========================
st.markdown("### 🌤️ Analisis Visual")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    if mode == "Deteksi Objek (YOLO)" and yolo_model:
        results = yolo_model(img)
        plotted = results[0].plot()
        st.image(plotted, caption="✨ Hasil Deteksi", use_container_width=True)

    elif mode == "Klasifikasi Gambar" and classifier:
        input_shape = classifier.input_shape[1:3]
        img_resized = img.resize(input_shape)
        img_array = image.img_to_array(img_resized)
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        conf = np.max(prediction)
        st.success(f"✅ Prediksi: **{class_index}** ({conf*100:.2f}%)")
    else:
        st.warning("⚠️ Model tidak ditemukan di folder Model/.")
else:
    st.info("🖼️ Silakan unggah gambar terlebih dahulu.")

# ==========================
# FOOTER
# ==========================
st.markdown("""
<div class='footer'>
© 2025 — HoloFruits Vision Dashboard | By Intan Humaira 💫
</div>
""", unsafe_allow_html=True)
