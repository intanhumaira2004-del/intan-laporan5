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
# 🎨 FINAL CSS DASHBOARD TEMA MAROON-ORANGE STATISTIKA SEGAR 🍊📊
<style>

/* ====== BACKGROUND MAROON KALEM BERGERAK ====== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #faf7f8 0%, #f1e1e3 40%, #e6ccd0 80%);
    background-size: 300% 300%;
    animation: gradientFlow 20s ease infinite;
    min-height: 100vh;
    overflow: auto;
}
@keyframes gradientFlow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* ====== TITIK STATISTIK HALUS ====== */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: absolute;
    inset: 0;
    background-image: radial-gradient(rgba(120, 40, 40, 0.08) 1px, transparent 1px);
    background-size: 28px 28px;
    animation: moveDots 30s linear infinite;
    z-index: 0;
}
@keyframes moveDots {
    0% {background-position: 0 0;}
    100% {background-position: 120px 120px;}
}

/* ====== IKON KECIL BERTEMA STATISTIKA ====== */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        url('https://cdn-icons-png.flaticon.com/512/686/686589.png'),
        url('https://cdn-icons-png.flaticon.com/512/4149/4149686.png'),
        url('https://cdn-icons-png.flaticon.com/512/2306/2306164.png');
    background-repeat: no-repeat;
    background-size: 150px, 140px, 160px;
    background-position: 8% 15%, 75% 25%, 60% 80%;
    opacity: 0.05;
    z-index: 0;
}

/* ====== KONTEN TIDAK TERPOTONG ====== */
.block-container {
    max-width: 1050px;
    margin: auto;
    padding-bottom: 60px;
    position: relative;
    z-index: 2;
}

/* ====== HEADER ====== */
.header {
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.7);
    padding: 18px 30px;
    border-radius: 20px;
    box-shadow: 0 4px 25px rgba(120,40,40,0.2);
    backdrop-filter: blur(10px);
    margin-bottom: 25px;
    border: 1px solid rgba(200,150,150,0.25);
    z-index: 2;
}
.header img {
    width: 130px;
    margin-right: 22px;
    filter: drop-shadow(0 0 12px rgba(160,70,70,0.4));
    animation: float 4s ease-in-out infinite;
}
@keyframes float {
    0%,100% {transform: translateY(0px);}
    50% {transform: translateY(-6px);}
}

/* ====== ANIMASI JUDUL BERGERAK ====== */
.title-text {
    font-size: 38px;
    font-weight: 800;
    background: linear-gradient(90deg, #5e0d0d, #9c3636, #d47c59);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 5s linear infinite;
    text-shadow: 0 0 25px rgba(160,60,60,0.25);
}
@keyframes shine {
    0% {background-position: 0%;}
    100% {background-position: 200%;}
}

/* ====== GLASS CARD ====== */
.glass-card {
    background: rgba(255,255,255,0.85);
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(180,100,100,0.3);
    box-shadow: 0 6px 22px rgba(120,40,40,0.1);
    backdrop-filter: blur(14px);
    margin-bottom: 25px;
}

/* ====== FOOTER ====== */
footer {
    text-align: center;
    color: #5e0d0d;
    margin-top: 45px;
    font-size: 14px;
    opacity: 0.8;
}
</style>

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
