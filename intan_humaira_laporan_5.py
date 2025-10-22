# -*- coding: utf-8 -*-
"""USK HoloVision Dashboard 🌈 Unik & Cerah"""

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import plotly.graph_objects as go

# ==========================
# Konfigurasi Dasar
# ==========================
st.set_page_config(page_title="USK HoloVision Dashboard", page_icon="🌈", layout="wide")

# ==========================
# CSS Unik & Cerah
# ==========================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #e0f7ff, #f8fbff, #ffffff);
    animation: gradientMove 10s ease infinite alternate;
}
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    100% {background-position: 100% 50%;}
}

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

# -------------------------
# HEADER (FIXED — NO DOUBLE LOGO)
# -------------------------
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
    <div class="header-container">
        <div>
            <div class="title-text">Neura HoloLab 3D — USK Statistics</div>
            <div class="subtitle">Faculty of Mathematics and Natural Sciences</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================
# Deskripsi Dataset 📚
# ==========================
st.markdown("""
<div class="glass-card">
    <h3>🧺 Deskripsi Dataset: <em>Fruits Fresh and Rotten for Classification</em></h3>
    <p style='text-align: justify;'>
    Dataset ini berasal dari platform 
    <a href='https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification' target='_blank'>Kaggle</a>. 
    Dataset ini berisi kumpulan gambar buah-buahan dalam dua kondisi, yaitu <b>fresh (segar)</b> dan <b>rotten (busuk)</b>, 
    mencakup tiga jenis buah: <b>apel, pisang, dan jeruk</b>. 
    Setiap kombinasi menghasilkan enam kelas gambar sebagai berikut:
    <ul>
        <li>freshapples 🍎</li>
        <li>rottenapples 🍏</li>
        <li>freshbanana 🍌</li>
        <li>rottenbanana 🍌</li>
        <li>freshoranges 🍊</li>
        <li>rottenoranges 🍊</li>
    </ul>
    Dataset terbagi menjadi dua bagian utama:
    <br>📁 <b>Train</b>: 10.901 gambar  
    <br>📁 <b>Test</b>: 2.698 gambar
    <br><br>
    Tujuan utama dataset ini adalah untuk melatih dan menguji model klasifikasi gambar 
    agar dapat mengenali kondisi buah berdasarkan penampilan visualnya. 
    Dataset ini banyak digunakan dalam penelitian bidang <b>Computer Vision</b> dan <b>Deep Learning</b> 
    menggunakan arsitektur <b>Convolutional Neural Network (CNN)</b>.
    </p>
</div>
""", unsafe_allow_html=True)
# ==========================
# Load model
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
# Sidebar Upload & Mode
# ==========================
st.sidebar.title("🎛️ Mode Analisis")
mode = st.sidebar.selectbox("Pilih Fungsi:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.sidebar.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# Konten Utama
# ==========================
st.markdown("### 🌤️ Analisis Visual Holografik")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    if mode == "Deteksi Objek (YOLO)" and yolo_model:
        results = yolo_model(img)
        plotted = results[0].plot()
        st.image(plotted, caption="✨ Hasil Deteksi", use_container_width=True)

        # Hitung jumlah objek
        obj_counts = {}
        for cls in results[0].boxes.cls:
            label = results[0].names[int(cls)]
            obj_counts[label] = obj_counts.get(label, 0) + 1

        if obj_counts:
            fig = go.Figure(go.Pie(
                labels=list(obj_counts.keys()),
                values=list(obj_counts.values()),
                hole=0.4,
                marker=dict(colors=["#00CFFF","#00AEEF","#89E1FF","#0088FF"])
            ))
            fig.update_layout(
                title="📊 Proporsi Objek yang Terdeteksi",
                font=dict(color="#005b96")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tidak ada objek terdeteksi.")

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
# Footer
# ==========================
st.markdown("""
<footer>
© 2025 — USK HoloVision Dashboard | Intan Humaira 💫
</footer>
""", unsafe_allow_html=True)
