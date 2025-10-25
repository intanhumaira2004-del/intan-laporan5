# -*- coding: utf-8 -*-
"""🌈 Dashboard Klasifikasi Buah Segar dan Busuk"""

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
st.set_page_config(page_title="Dashboard Klasifikasi Buah Segar dan Busuk", page_icon="🍎", layout="wide")

# ==========================
# CSS STYLING DASHBOARD 🌈
# ==========================
st.markdown("""
<style>
/* =====================
🌈 BACKGROUND & DEKORASI
===================== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e0f7ff 0%, #f8fcff 35%, #ffffff 100%);
    background-attachment: fixed;
    background-size: 200% 200%;
    animation: gradientShift 12s ease infinite;
    min-height: 100vh;
    position: relative;
    overflow: hidden;
}

@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Dekorasi halus kiri atas */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 40px;
    left: 40px;
    width: 220px;
    height: 220px;
    background: url('https://cdn-icons-png.flaticon.com/512/7660/7660618.png') no-repeat center;
    background-size: 180px;
    opacity: 0.15;
    filter: drop-shadow(0 0 10px rgba(0,150,255,0.3));
}

/* Dekorasi halus kanan bawah */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: absolute;
    bottom: 40px;
    right: 40px;
    width: 260px;
    height: 260px;
    background: url('https://cdn-icons-png.flaticon.com/512/2909/2909769.png') no-repeat center;
    background-size: 200px;
    opacity: 0.12;
    filter: drop-shadow(0 0 8px rgba(0,120,255,0.25));
}

/* =====================
✨ HEADER STYLE
===================== */
.header {
    display:flex;
    align-items:center;
    justify-content:center;
    background: rgba(255,255,255,0.7);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 4px 25px rgba(0,150,255,0.2);
    backdrop-filter: blur(12px);
    margin-bottom: 25px;
}

.header img {
    width: 100px;
    margin-right: 20px;
    filter: drop-shadow(0 0 15px rgba(0,200,255,0.5));
    animation: float 4s ease-in-out infinite;
}
@keyframes float {
    0%,100% {transform: translateY(0px);}
    50% {transform: translateY(-6px);}
}

/* =====================
🎨 TEKS & KOMPONEN
===================== */
.title-text {
    font-size: 34px;
    font-weight: 800;
    background: linear-gradient(90deg,#00aaff,#00e1ff,#0088ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 25px rgba(0,180,255,0.4);
}

.glass-card {
    background: rgba(255,255,255,0.75);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(180,220,255,0.4);
    box-shadow: 0 6px 20px rgba(0,100,200,0.15);
    backdrop-filter: blur(12px);
}

footer {
    text-align:center;
    color:#0080b9;
    margin-top:40px;
    font-size:14px;
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
        st.markdown("<div style='width:90px;height:90px;background:#0b2149;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#9fd7ff;font-weight:700;'>USK</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="header">
        <div class="title-text">🍎 Dashboard Klasifikasi Buah Segar dan Busuk 🍌</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================
# DESKRIPSI DATASET 📚
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
    yolo_path = "Model/Intan Humaira_Laporan 4.pt"
    keras_path = "Model/Intan Humaira_Laporan2.h5"

    yolo_model = YOLO(yolo_path) if os.path.exists(yolo_path) else None
    classifier = tf.keras.models.load_model(keras_path) if os.path.exists(keras_path) else None
    return yolo_model, classifier

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
st.markdown("### 🌤️ Analisis Visual Gambar Buah")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    # ==========================
    # MODE DETEKSI YOLO
    # ==========================
    if mode == "Deteksi Objek (YOLO)" and yolo_model:
        results = yolo_model(img)
        plotted = results[0].plot()
        st.image(plotted, caption="✨ Hasil Deteksi", use_container_width=True)

    # ==========================
    # MODE KLASIFIKASI GAMBAR
    # ==========================
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
<footer>
© 2025 — Dashboard Klasifikasi Buah Segar dan Busuk | Intan Humaira 💫
</footer>
""", unsafe_allow_html=True)
