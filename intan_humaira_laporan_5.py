# -*- coding: utf-8 -*-
"""INTAN HUMAIRA_DASHBOARD_UNIK"""

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Konfigurasi Dasar
# ==========================
st.set_page_config(
    page_title="VisionX - Image Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("Model/Intan Humaira_Laporan 4.pt")
    classifier = tf.keras.models.load_model("Model/Intan Humaira_Laporan2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Header
# ==========================
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        color: #6A1B9A;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #444;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<p class="main-title">✨ VisionX: Smart Image Analyzer ✨</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Deteksi objek & klasifikasi gambar dengan model Intan Humaira</p>', unsafe_allow_html=True)

# ==========================
# Sidebar
# ==========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4359/4359957.png", width=130)
    st.header("🔍 Pilihan Mode")
    mode = st.radio("Pilih Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

    st.markdown("---")
    st.info("Unggah gambar berformat JPG, JPEG, atau PNG untuk memulai analisis.")
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# Main Content
# ==========================
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Gambar yang Diupload", use_container_width=True)

    tab1, tab2 = st.tabs(["📦 Hasil Analisis", "📊 Detail Prediksi"])

    with tab1:
        if mode == "Deteksi Objek (YOLO)":
            st.subheader("🧩 Hasil Deteksi Objek")
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="Gambar dengan Bounding Box", use_container_width=True)
            st.success("✅ Deteksi selesai! Objek berhasil diidentifikasi.")

        elif mode == "Klasifikasi Gambar":
            st.subheader("🎯 Hasil Klasifikasi")
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = float(np.max(prediction))

            st.markdown(f"### 🧠 Prediksi Kelas: `{class_index}`")
            st.progress(confidence)
            st.write(f"**Probabilitas:** {confidence:.2%}")

    with tab2:
        st.write("📈 Grafik atau visualisasi tambahan dapat ditambahkan di sini, seperti:")
        st.markdown("- Distribusi kelas prediksi")
        st.markdown("- Confidence chart")
        st.markdown("- Riwayat hasil deteksi")

else:
    st.info("Silakan unggah gambar terlebih dahulu dari sidebar kiri untuk memulai analisis.")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 VisionX | Dikembangkan oleh <b>Intan Humaira</b></p>",
    unsafe_allow_html=True
)
