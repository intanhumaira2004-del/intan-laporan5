# -*- coding: utf-8 -*-
"""HoloVision Dashboard - Maroon Theme"""

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
# CSS STYLING (SOFT ROSE GLOW THEME)
# ==========================
# 🌸 Background + Desain Visual Statistik Unik & Elegan
st.markdown("""
<style>
/* ====== BACKGROUND DASHBOARD ====== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f8e7e7 0%, #f0d6d6 40%, #e8c4c4 75%, #dba9a9 100%);
    background-attachment: fixed;
    position: relative;
    overflow: hidden;
}

/* Efek cahaya lembut */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: -200px;
    left: -200px;
    width: 900px;
    height: 900px;
    background: radial-gradient(circle, rgba(255,255,255,0.45), transparent 70%);
    filter: blur(160px);
    z-index: 0;
}

/* Pola statistik halus */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        radial-gradient(circle at 25% 35%, rgba(255,255,255,0.06) 0%, transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(255,255,255,0.05) 0%, transparent 50%),
        repeating-linear-gradient(
            135deg,
            rgba(255,255,255,0.06) 0px,
            rgba(255,255,255,0.06) 1px,
            transparent 2px,
            transparent 6px
        ),
        url('https://www.transparenttextures.com/patterns/dot-grid.png');
    background-size: 350px 350px, 350px 350px, 400px 400px, 250px 250px;
    opacity: 0.9;
    z-index: 0;
}

/* ====== WAVE EFEK ====== */
.wave-top, .wave-bottom {
    position: fixed;
    left: 0;
    width: 100%;
    height: 180px;
    background-repeat: no-repeat;
    background-size: cover;
    opacity: 0.3;
    z-index: 0;
}
.wave-top {
    top: 0;
    background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320"><path fill="%23e4b4b4" fill-opacity="1" d="M0,192L80,170.7C160,149,320,107,480,96C640,85,800,107,960,144C1120,181,1280,235,1360,261.3L1440,288L1440,0L1360,0C1280,0,1120,0,960,0C800,0,640,0,480,0C320,0,160,0,80,0L0,0Z"></path></svg>');
}
.wave-bottom {
    bottom: 0;
    background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320"><path fill="%23e4b4b4" fill-opacity="1" d="M0,160L80,138.7C160,117,320,75,480,90.7C640,107,800,181,960,213.3C1120,245,1280,235,1360,229.3L1440,224L1440,320L1360,320C1280,320,1120,320,960,320C800,320,640,320,480,320C320,320,160,320,80,320L0,320Z"></path></svg>');
}

/* ====== WATERMARK STATISTIK ====== */
.stat-shape {
    position: fixed;
    opacity: 0.07;
    z-index: 0;
    transform: rotate(-10deg);
}
.stat-shape.chart-left {
    top: 5%;
    left: 2%;
    width: 260px;
    height: 260px;
    background: url('https://cdn-icons-png.flaticon.com/512/2830/2830310.png') no-repeat center;
    background-size: contain;
}
.stat-shape.chart-right {
    bottom: 5%;
    right: 2%;
    width: 280px;
    height: 280px;
    background: url('https://cdn-icons-png.flaticon.com/512/3225/3225193.png') no-repeat center;
    background-size: contain;
}

/* ====== CARD (GLASSMORPHISM) ====== */
div[data-testid="stMarkdownContainer"] .glass-card {
    background: rgba(255, 255, 255, 0.75);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    backdrop-filter: blur(12px);
}

/* ====== PASTIKAN KONTEN DI DEPAN ====== */
header, footer, .stApp, .block-container {
    position: relative;
    z-index: 1;
}
</style>

<div class="wave-top"></div>
<div class="wave-bottom"></div>
<div class="stat-shape chart-left"></div>
<div class="stat-shape chart-right"></div>
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
    st.markdown("""
    <div class="header">
        <div class="title-text">HoloFruits Vision Dashboard</div>
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
<footer>
© 2025 — HoloFruits Vision Dashboard | By Intan Humaira 💫
</footer>
""", unsafe_allow_html=True)
