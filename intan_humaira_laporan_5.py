# -*- coding: utf-8 -*-
"""Neura HoloLab 3D - Dashboard Cerah"""

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
st.set_page_config(page_title="Neura HoloLab 3D", page_icon="✨", layout="wide")

# ==========================
# Gaya CSS Cerah (Sky Neon Light)
# ==========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #e9f7ff 0%, #f9fcff 50%, #e1f3ff 100%);
    color: #023047;
    font-family: "Poppins", sans-serif;
}

/* Header elegan */
.header-card {
    display:flex;
    align-items:center;
    gap:20px;
    padding:18px 24px;
    border-radius:18px;
    background: linear-gradient(120deg, rgba(240,250,255,1), rgba(210,235,255,0.95));
    border: 1px solid rgba(160,210,255,0.7);
    box-shadow: 0 0 30px rgba(180,220,255,0.5);
    backdrop-filter: blur(10px);
}

/* Logo glowing */
.usk-logo {
    width:85px;
    height:auto;
    border-radius:12px;
    box-shadow: 0 0 20px rgba(0,170,255,0.6);
    animation: glowspin 10s infinite ease-in-out;
}

@keyframes glowspin {
    0%,100% { transform: rotate(0deg); box-shadow:0 0 18px rgba(0,180,255,0.6); }
    50% { transform: rotate(5deg); box-shadow:0 0 30px rgba(0,220,255,0.7); }
}

/* Judul teks */
.site-title {
    font-size:30px;
    font-weight:800;
    background: linear-gradient(90deg,#007BFF,#00CFFF,#00E1FF);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    text-shadow: 0 0 20px rgba(150,220,255,0.6);
    animation: shimmer 6s ease-in-out infinite;
}

@keyframes shimmer {
    0%,100% { text-shadow:0 0 15px rgba(0,180,255,0.4), 0 0 30px rgba(0,220,255,0.3); }
    50% { text-shadow:0 0 25px rgba(0,230,255,0.6), 0 0 40px rgba(120,240,255,0.5); }
}

/* Konten */
.section-card {
    background: rgba(255,255,255,0.75);
    border-radius:16px;
    padding:20px;
    border: 1px solid rgba(180,220,255,0.6);
    box-shadow: 0 6px 20px rgba(0,80,160,0.15);
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Header dengan Logo USK
# ==========================
st.markdown("""
<div class="header-card">
    <img src="https://upload.wikimedia.org/wikipedia/id/2/29/Logo_Unsyiah.png" class="usk-logo">
    <div>
        <div class="site-title">Neura HoloLab 3D</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================
# Fungsi Load Model (cache)
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
st.sidebar.title("🔧 Mode")
mode = st.sidebar.radio("Pilih Fungsi:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.sidebar.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# Konten Utama
# ==========================
st.markdown("### 🌤️ Analisis Gambar")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    if mode == "Deteksi Objek (YOLO)" and yolo_model:
        results = yolo_model(img)
        plotted = results[0].plot()
        st.image(plotted, caption="Hasil Deteksi", use_container_width=True)

        # Buat chart hasil deteksi
        obj_counts = {}
        for cls in results[0].boxes.cls:
            label = results[0].names[int(cls)]
            obj_counts[label] = obj_counts.get(label, 0) + 1

        if obj_counts:
            fig = go.Figure(go.Bar(
                x=list(obj_counts.keys()),
                y=list(obj_counts.values()),
                marker_color="#00AEEF"
            ))
            fig.update_layout(
                title="📊 Jumlah Objek Terdeteksi",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#004c91")
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

        st.success(f"✅ Kelas Prediksi: **{class_index}** ({conf*100:.2f}%)")
    else:
        st.warning("⚠️ Model tidak ditemukan. Pastikan file model sudah diunggah ke folder Model/.")
else:
    st.info("🖼️ Silakan unggah gambar terlebih dahulu.")

# ==========================
# Footer
# ==========================
st.markdown("""
<hr style="border:1px solid rgba(0,120,255,0.2);margin-top:50px;">
<div style="text-align:center;color:#005b96;font-size:14px;">
© 2025 — Neura HoloLab 3D | Universitas Syiah Kuala 🌤️
</div>
""", unsafe_allow_html=True)
