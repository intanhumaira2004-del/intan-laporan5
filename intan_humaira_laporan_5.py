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
# CSS STYLING (Tema Statistik Modern + Background Bergambar)
# ==========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: 
        linear-gradient(rgba(240, 247, 255, 0.92), rgba(240, 247, 255, 0.92)),
        url("2D92E412-53C3-4776-9179-5F99A0176B7D.jpeg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
}

/* Header dengan efek neon lembut */
.header {
    display:flex;
    align-items:center;
    justify-content:center;
    background: rgba(255, 255, 255, 0.8);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 4px 25px rgba(0, 80, 180, 0.25);
    backdrop-filter: blur(10px);
    margin-bottom: 25px;
}
.header img { 
    width: 95px; 
    margin-right: 20px; 
    filter: drop-shadow(0 0 10px rgba(0, 150, 255, 0.5)); 
    animation: float 4s ease-in-out infinite; 
}
@keyframes float { 
    0%,100% {transform: translateY(0px);} 
    50% {transform: translateY(-6px);} 
}

/* Judul utama dengan gradasi futuristik */
.title-text {
    font-size: 34px;
    font-weight: 800;
    background: linear-gradient(90deg,#004aad,#5e60ce,#00b4d8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 18px rgba(0,120,255,0.3);
}

/* Kartu kaca untuk konten */
.glass-card {
    background: rgba(255,255,255,0.75);
    border-radius: 16px;
    padding: 22px;
    border: 1px solid rgba(160,200,255,0.4);
    box-shadow: 0 6px 20px rgba(0,100,200,0.15);
    backdrop-filter: blur(12px);
}

/* Tombol sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #004aad 0%, #5e60ce 100%);
}
.stButton>button {
    background-color: #4ea8de !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 12px rgba(0,80,160,0.3);
}
.stButton>button:hover {
    background-color: #4361ee !important;
}

/* Footer */
footer {
    text-align:center;
    color:#003366;
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
        <span style='font-size:18px;font-weight:500;color:#0056b3;'>A Statistical Approach to AI-Based Fruit Classification</span></div>
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
