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
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="USK HoloVision Dashboard", page_icon="🌈", layout="wide")

# ==========================
# CSS STYLING DASHBOARD 🌈
# ==========================
st.markdown("""
<style>
/* =====================
🌈 BACKGROUND DASHBOARD FIX
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

/* Efek gradasi dinamis */
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Dekorasi halus */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: -80px;
    left: -120px;
    width: 800px;
    height: 800px;
    background: url('https://cdn-icons-png.flaticon.com/512/6062/6062646.png') no-repeat;
    background-size: 320px;
    opacity: 0.08;
    transform: rotate(25deg);
}

[data-testid="stAppViewContainer"]::after {
    content: "";
    position: absolute;
    bottom: -100px;
    right: -120px;
    width: 900px;
    height: 900px;
    background: url('https://cdn-icons-png.flaticon.com/512/4149/4149676.png') no-repeat;
    background-size: 340px;
    opacity: 0.09;
    transform: rotate(-15deg);
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
        <div class="title-text">Neura HoloLab 3D</div>
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
st.markdown("### 🌤️ Analisis Visual Holografik")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    if mode == "Deteksi Objek (YOLO)" and yolo_model:
        results = yolo_model(img)
        plotted = results[0].plot()
        st.image(plotted, caption="✨ Hasil Deteksi", use_container_width=True)

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
# FOOTER
# ==========================
st.markdown("""
<footer>
© 2025 — HoloVision Dashboard | Intan Humaira 💫
</footer>
""", unsafe_allow_html=True)
