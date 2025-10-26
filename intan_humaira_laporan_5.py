# -*- coding: utf-8 -*-
"""HoloVision Dashboard"""

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import plotly.graph_objects as go

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="HoloFruits Vision Dashboard", layout="wide")
# ==========================
st.markdown("""
<style>

/* 🌈 Background utama dengan efek gerak halus */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #fff5f7, #ffeef1, #fbe7ea, #ffeef5);
    background-size: 300% 300%;
    animation: gradientMove 18s ease-in-out infinite;
    overflow-y: auto !important;
}

/* ✨ Animasi pergeseran warna */
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* 🍓 Pola lembut titik-titik */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    inset: 0;
    background-image: radial-gradient(rgba(255, 100, 130, 0.05) 1px, transparent 1px);
    background-size: 40px 40px;
    z-index: 0;
}

/* 🍊 Ikon kecil buah dan grafik yang bergerak pelan */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        url('https://cdn-icons-png.flaticon.com/512/415/415733.png'),
        url('https://cdn-icons-png.flaticon.com/512/4149/4149676.png'),
        url('https://cdn-icons-png.flaticon.com/512/135/135620.png');
    background-repeat: no-repeat;
    background-size: 70px, 80px, 90px;
    background-position: 5% 10%, 90% 60%, 50% 90%;
    opacity: 0.08;
    animation: floatIcons 25s linear infinite;
    pointer-events: none;
    z-index: 0;
}

/* 🍉 Animasi lembut untuk ikon latar */
@keyframes floatIcons {
    0% { background-position: 5% 10%, 90% 60%, 50% 90%; }
    50% { background-position: 7% 15%, 88% 55%, 48% 92%; }
    100% { background-position: 5% 10%, 90% 60%, 50% 90%; }
}

/* 📏 Batas lebar dashboard */
[data-testid="stVerticalBlock"] {
    max-width: 1050px;
    margin: auto;
    padding: 1rem 2rem;
    position: relative;
    z-index: 2;
}

/* 🪞 Header dan logo */
.header {
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.8);
    padding: 20px 30px;
    border-radius: 22px;
    box-shadow: 0 4px 25px rgba(150,0,0,0.1);
    backdrop-filter: blur(10px);
    margin-bottom: 25px;
    z-index: 2;
}
.header img {
    width: 220px; /* ✅ Diperbesar dari sebelumnya */
    margin-right: 25px;
    filter: drop-shadow(0 0 8px rgba(120,0,0,0.25));
    transition: transform 0.3s ease-in-out;
}
.header img:hover {
    transform: scale(1.05); /* ✨ Efek hover lembut */
}

/* 🩷 Judul dashboard dengan efek gerak & glow */
.title-text {
    font-size: 36px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg,#8b1e1e,#c94b4b,#f28f8f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glowText 6s ease-in-out infinite, floatTitle 4s ease-in-out infinite;
}
@keyframes glowText {
    0%,100% { text-shadow: 0 0 10px rgba(180,50,50,0.3); }
    50% { text-shadow: 0 0 25px rgba(220,70,70,0.6); }
}
@keyframes floatTitle {
    0%,100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}

/* 🍋 Kartu konten */
.glass-card {
    background: rgba(255,255,255,0.85);
    border-radius: 18px;
    padding: 22px;
    border: 1px solid rgba(200,150,150,0.3);
    box-shadow: 0 4px 18px rgba(120,0,0,0.1);
    backdrop-filter: blur(14px);
    z-index: 2;
}

/* 🍍 Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(8px);
    border-right: 1px solid rgba(160,80,80,0.15);
}

/* 📊 Footer */
footer {
    text-align: center;
    color: #7a1c1c;
    margin-top: 40px;
    font-size: 14px;
    font-weight: 500;
}

html {
    scroll-behavior: smooth;
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
        st.markdown("<div style='width:90px;height:90px;background:#7a1c1c;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#ffeaea;font-weight:700;'>USK</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="header">
        <div class="title-text">HoloFruits Vision Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

# Tambahkan elemen visual statistik (ikon grafik lembut)
st.markdown("""
<img class='stat-deco' src='https://cdn-icons-png.flaticon.com/512/4149/4149686.png'>
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
st.markdown("### 🌤️ Analisis Visual Holografik")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    if mode == "Deteksi Objek (YOLO)" and yolo_model:
        results = yolo_model(img)
        plotted = results[0].plot()
        st.image(plotted, caption="✨ Hasil Deteksi", use_container_width=True)

    elif mode == "Klasifikasi Gambar" and classifier:
        st.write("Ukuran input model:", classifier.input_shape)
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
© 2025 — HoloFruits Vision Dashboard | Intan Humaira 💫
</footer>
""", unsafe_allow_html=True)
