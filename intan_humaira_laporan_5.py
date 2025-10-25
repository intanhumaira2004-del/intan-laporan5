# -*- coding: utf-8 -*-
"""FruitVision Dashboard 🍉 Fresh vs Rotten"""

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
st.set_page_config(page_title="FruitVision Dashboard 🍉", page_icon="🍓", layout="wide")

# ==========================
# CSS STYLING DASHBOARD 🍉
# ==========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #fff8f3 0%, #f8fcff 40%, #ffffff 100%);
    background-attachment: fixed;
    background-size: 200% 200%;
    animation: gradientShift 14s ease infinite;
    min-height: 100vh;
    position: relative;
    overflow: hidden;
}
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: -100px;
    left: -150px;
    width: 700px;
    height: 700px;
    background: url('https://cdn-icons-png.flaticon.com/512/415/415682.png') no-repeat;
    background-size: 320px;
    opacity: 0.07;
    transform: rotate(15deg);
}
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: absolute;
    bottom: -100px;
    right: -150px;
    width: 700px;
    height: 700px;
    background: url('https://cdn-icons-png.flaticon.com/512/706/706164.png') no-repeat;
    background-size: 320px;
    opacity: 0.07;
    transform: rotate(-20deg);
}
.header {
    display:flex;
    align-items:center;
    justify-content:center;
    background: rgba(255,255,255,0.7);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 4px 25px rgba(255,150,0,0.2);
    backdrop-filter: blur(12px);
    margin-bottom: 25px;
}
.header img {
    width: 100px;
    margin-right: 20px;
    filter: drop-shadow(0 0 15px rgba(255,180,0,0.5));
    animation: float 4s ease-in-out infinite;
}
@keyframes float {
    0%,100% {transform: translateY(0px);}
    50% {transform: translateY(-6px);}
}
.title-text {
    font-size: 34px;
    font-weight: 800;
    background: linear-gradient(90deg,#ff6600,#ffb84d,#ff884d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 25px rgba(255,150,0,0.3);
}
.glass-card {
    background: rgba(255,255,255,0.75);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,200,150,0.4);
    box-shadow: 0 6px 20px rgba(255,150,50,0.15);
    backdrop-filter: blur(12px);
}
footer {
    text-align:center;
    color:#ff7700;
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
    "assets/fruit_logo.png",
    "usk_logo.png"
]
logo_path = next((p for p in logo_candidates if os.path.exists(p)), None)

col1, col2 = st.columns([0.15, 0.85])
with col1:
    if logo_path:
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown("<div style='width:90px;height:90px;background:#ff7700;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;'>🍊</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="header">
        <div class="title-text">FruitVision Dashboard 🍉 Fresh vs Rotten</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================
# DESKRIPSI DATASET 🍎
# ==========================
st.markdown("""
<div class="glass-card">
    <h3>📚 Deskripsi Dataset: <em>Fruits Fresh and Rotten for Classification</em></h3>
    <p style='text-align: justify;'>
    Dataset ini berasal dari 
    <a href='https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification' target='_blank'>Kaggle</a>. 
    Dataset ini berisi gambar buah-buahan dalam dua kondisi, yaitu <b>fresh</b> (segar) dan <b>rotten</b> (busuk),
    mencakup tiga jenis buah utama: apel, pisang, dan jeruk.
    <br><br>
    Terdapat enam kelas gambar:
    <ul>
        <li>🍎 freshapples</li>
        <li>🍏 rottenapples</li>
        <li>🍌 freshbanana</li>
        <li>🍌 rottenbanana</li>
        <li>🍊 freshoranges</li>
        <li>🍊 rottenoranges</li>
    </ul>
    Dataset ini banyak digunakan dalam riset <b>Deep Learning</b> untuk membangun model klasifikasi berbasis <b>Convolutional Neural Network (CNN)</b>.
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
st.markdown("### 🍋 Analisis Visual Buah Segar & Busuk")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    # ==========================
    # MODE DETEKSI YOLO
    # ==========================
    if mode == "Deteksi Objek (YOLO)" and yolo_model:
        results = yolo_model(img)
        plotted = results[0].plot()
        st.image(plotted, caption="✨ Hasil Deteksi Objek", use_container_width=True)

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

        # Tambahan gambar ilustrasi dekoratif
        st.image("https://cdn-icons-png.flaticon.com/512/4149/4149676.png", 
                 caption="Ilustrasi Grafik (Hanya untuk Desain Tampilan)", 
                 use_container_width=True)

    else:
        st.warning("⚠️ Model tidak ditemukan di folder Model/.")
else:
    st.info("🖼️ Silakan unggah gambar terlebih dahulu.")

# ==========================
# FOOTER
# ==========================
st.markdown("""
<footer>
© 2025 — FruitVision Dashboard 🍉 | Intan Humaira 💫 <br>
<i>Ilustrasi grafik bersifat dekoratif, bukan hasil analisis model.</i>
</footer>
""", unsafe_allow_html=True)
