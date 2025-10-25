# -*- coding: utf-8 -*-
"""HoloFruit Vision Dashboard"""
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
st.set_page_config(page_title="HoloFruit Vision Dashboard", layout="wide")

# ==========================
# CSS STYLING (Tema Statistik + Gradasi Elegan)
# ==========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg, 
        rgba(240,248,255,0.93) 0%, 
        rgba(255,255,255,0.96) 40%, 
        rgba(245,250,255,1) 100%
    ),
    url("https://cdn.pixabay.com/photo/2020/09/29/16/51/data-5615376_1280.jpg");
    background-repeat: no-repeat;
    background-position: center;
    background-size: cover;
    background-attachment: fixed;
}

/* Efek lembut di konten */
.header {
    display:flex;
    align-items:center;
    justify-content:center;
    background: rgba(255,255,255,0.7);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 4px 25px rgba(0,100,200,0.15);
    backdrop-filter: blur(10px);
    margin-bottom: 25px;
}
.header img { width: 100px; margin-right: 20px; filter: drop-shadow(0 0 15px rgba(0,150,255,0.5)); animation: float 4s ease-in-out infinite; }
@keyframes float { 0%,100% {transform: translateY(0px);} 50% {transform: translateY(-6px);} }

.title-text {
    font-size: 34px;
    font-weight: 800;
    background: linear-gradient(90deg,#007bff,#00d4ff,#004aad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 20px rgba(0,160,255,0.3);
}

.glass-card {
    background: rgba(255,255,255,0.8);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(160,200,255,0.35);
    box-shadow: 0 6px 20px rgba(0,100,200,0.12);
    backdrop-filter: blur(10px);
}
footer {
    text-align:center;
    color:#004aad;
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
        <div class="title-text">HoloFruit Vision Dashboard 🍎<br>
        <span style='font-size:18px;font-weight:500;color:#007bff;'>A Statistical Approach to AI-Based Fruit Classification</span></div>
    </div>
    """, unsafe_allow_html=True)

# ==========================
# DESKRIPSI DATASET (tidak diubah)
# ==========================
st.markdown("""
<div class="glass-card">
<h3>📊 Deskripsi Dataset: <em>Fruits Fresh and Rotten for Classification</em></h3>
<p style='text-align: justify;'>
Dataset ini berasal dari platform <a href='https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification' target='_blank'>Kaggle</a>.
Dataset ini berisi gambar buah-buahan dalam dua kondisi: fresh (segar) dan rotten (busuk),
mencakup tiga jenis buah — apel, pisang, dan jeruk. Total terdapat enam kelas gambar:
<li>freshapples</li>
<li>rottenapples</li>
<li>freshbanana</li>
<li>rottenbanana</li>
<li>freshoranges</li>
<li>rottenoranges</li>
Dataset ini digunakan untuk melatih model CNN agar mampu mengenali kondisi buah secara otomatis.
Dataset ini juga relevan untuk penelitian di bidang <b>Statistika Terapan, Computer Vision</b>, dan <b>Machine Learning</b>.
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
st.markdown("### 🔍 Analisis Visual Statistik & AI")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    if mode == "Deteksi Objek (YOLO)" and yolo_model:
        results = yolo_model(img)
        plotted = results[0].plot()
        st.image(plotted, caption="✨ Hasil Deteksi", use_container_width=True)

    elif mode == "Klasifikasi Gambar" and classifier:
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
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
© 2025 — HoloFruit Vision Dashboard | Created by Intan Humaira 
</footer>
""", unsafe_allow_html=True)
