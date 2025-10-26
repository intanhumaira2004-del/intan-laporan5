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

/* === BACKGROUND GRADIENT MAROON HOLOGRAPHIC === */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(125deg, #fff6f9 0%, #ffe9ef 30%, #f6d4da 60%, #e0b7c6 100%);
    background-attachment: fixed;
    background-size: 300% 300%;
    animation: holoShift 16s ease infinite;
    overflow: visible !important;
    position: relative !important;
    height: auto;
    min-height: 100vh !important;
    z-index: 0 !important;
}
@keyframes holoShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* === FIX SCROLL STREAMLIT === */
html, body, [data-testid="stAppViewContainer"], .main, .block-container {
    height: auto !important;
    min-height: 100vh !important;
    overflow-y: visible !important;
    overflow-x: hidden !important;
}
html {
    overflow-y: scroll !important;
    scroll-behavior: smooth;
}

/* === DEKORASI HOLOGRAFIK === */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.35), transparent 60%),
                radial-gradient(circle at 70% 40%, rgba(255,182,193,0.25), transparent 60%),
                radial-gradient(circle at 50% 90%, rgba(255,192,203,0.3), transparent 70%);
    animation: glowFloat 20s ease-in-out infinite alternate;
    z-index: 0;
    pointer-events: none;
}
@keyframes glowFloat {
    0% { transform: translateY(0px); opacity: 0.8; }
    50% { transform: translateY(-15px); opacity: 0.6; }
    100% { transform: translateY(0px); opacity: 0.8; }
}

/* === LAPISAN IKON === */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        url('https://cdn-icons-png.flaticon.com/512/415/415733.png'),
        url('https://cdn-icons-png.flaticon.com/512/766/766514.png'),
        url('https://cdn-icons-png.flaticon.com/512/135/135620.png'),
        url('https://cdn-icons-png.flaticon.com/512/4149/4149676.png');
    background-repeat: no-repeat;
    background-size: 160px, 140px, 150px, 180px;
    background-position: 10% 15%, 80% 20%, 15% 80%, 70% 70%;
    opacity: 0.08;
    animation: floatIcons 30s linear infinite;
    z-index: 0;
}
@keyframes floatIcons {
    0% { background-position: 10% 15%, 80% 20%, 15% 80%, 70% 70%; }
    50% { background-position: 12% 18%, 78% 25%, 17% 82%, 68% 73%; }
    100% { background-position: 10% 15%, 80% 20%, 15% 80%, 70% 70%; }
}

/* === HEADER ELEGAN === */
.header {
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.65);
    padding: 20px;
    border-radius: 22px;
    border: 1px solid rgba(150,0,0,0.2);
    box-shadow: 0 6px 25px rgba(100,0,0,0.15);
    backdrop-filter: blur(16px);
    margin-bottom: 25px;
    position: relative;
    overflow: hidden;
    z-index: 2;
}
.header::after {
    content: "";
    position: absolute;
    top: 0;
    left: -40%;
    width: 200%;
    height: 100%;
    background: linear-gradient(120deg, rgba(255,255,255,0.2), transparent, rgba(255,255,255,0.1));
    transform: skewX(-20deg);
    animation: lightSweep 6s linear infinite;
}
@keyframes lightSweep {
    0% { transform: translateX(-100%) skewX(-20deg); }
    100% { transform: translateX(100%) skewX(-20deg); }
}

/* === TEKS JUDUL === */
.title-text {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(90deg, #7a1f1f, #d96b6b, #ffb6c1, #ffc6c9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 25px rgba(255,150,150,0.25);
}

/* === CARD KACA === */
.glass-card {
    background: rgba(255,255,255,0.78);
    border-radius: 20px;
    padding: 24px;
    border: 1px solid rgba(190,120,120,0.3);
    box-shadow: 0 6px 22px rgba(100,0,0,0.15);
    backdrop-filter: blur(16px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(120,0,0,0.25);
}

/* === FOOTER === */
footer {
    text-align: center;
    color: #6a1a1a;
    margin-top: 45px;
    font-size: 14px;
    opacity: 0.8;
}

/* === STATISTIK HOLOGRAM === */
.stat-deco {
    position: absolute;
    top: 60px;
    right: 50px;
    opacity: 0.12;
    width: 230px;
    filter: drop-shadow(0 0 12px rgba(255,200,200,0.4));
    animation: rotateHolo 18s linear infinite;
}
@keyframes rotateHolo {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* === SIDEBAR === */
[data-testid="stSidebar"] {
    background: linear-gradient(145deg, #fff5f7 0%, #ffe0e6 35%, #f6ccd3 70%, #e8b6c3 100%) !important;
    color: #4a0f16 !important;
    backdrop-filter: blur(14px);
    border-right: 2px solid rgba(150, 0, 0, 0.1);
    box-shadow: 4px 0 20px rgba(150, 0, 0, 0.05);
}

/* === TEKS & UPLOADER SIDEBAR === */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #7a1f1f !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stFileUploader {
    background: rgba(255,255,255,0.75) !important;
    border-radius: 14px !important;
    padding: 10px !important;
    border: 1px solid rgba(180,80,90,0.25) !important;
    box-shadow: 0 4px 10px rgba(160,50,50,0.1) !important;
}
[data-testid="stSidebar"] .stFileUploader label,
[data-testid="stSidebar"] .stFileUploader div,
[data-testid="stSidebar"] .stFileUploader p {
    color: #2c0a0f !important;  
    font-weight: 500 !important;
}

/* === TOMBOL SIDEBAR === */
[data-testid="stSidebar"] button {
    background: linear-gradient(90deg, #b64b5a, #e7a2a9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(160, 60, 70, 0.25) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stSidebar"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(160, 60, 70, 0.3) !important;
}

/* === FIX AKHIR: PAKSA SCROLL CONTAINER STREAMLIT === */
main, .main, .block-container {
    overflow: visible !important;
    height: auto !important;
    min-height: 100vh !important;
}

[data-testid="stAppViewContainer"] > section {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    min-height: 100vh !important;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}

/* === Tambahkan ruang bawah ekstra === */
.block-container::after {
    content: "";
    display: block;
    height: 150px;
}

/* pastikan semua container bisa tumbuh dinamis */
html, body {
    height: auto !important;
    min-height: 100vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    scroll-behavior: smooth !important;
}

/* kontainer utama Streamlit */
[data-testid="stAppViewContainer"] {
    height: auto !important;
    min-height: 100vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    display: block !important;
}

/* kontainer utama tempat elemen muncul */
main, .main, .block-container {
    height: auto !important;
    min-height: 100vh !important;
    overflow-y: visible !important;
    padding-bottom: 150px !important; /* ruang ekstra bawah agar footer terlihat */
}

/* section internal */
[data-testid="stAppViewContainer"] > section {
    height: auto !important;
    min-height: 100vh !important;
    overflow-y: auto !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
}

/* perbaikan tambahan kalau masih ketarik ke atas */
.block-container::after {
    content: "";
    display: block;
    height: 150px; /* jarak aman bawah */
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
© 2025 — HoloFruits Vision Dashboard | By Intan Humaira 
</footer>
""", unsafe_allow_html=True)
