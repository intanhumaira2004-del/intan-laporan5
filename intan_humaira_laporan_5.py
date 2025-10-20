# -*- coding: utf-8 -*-
"""INTAN_HUMAIRA_NEURA_VISION.py"""

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import base64
import random

# ====================================
# PAGE CONFIG
# ====================================
st.set_page_config(
    page_title="NeuraVision - AI Futuristic Dashboard",
    page_icon="🤖",
    layout="wide"
)

# ====================================
# CUSTOM CSS
# ====================================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at 20% 20%, #090909, #1e0033);
    color: #eee;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3 {
    text-align: center;
    color: #A566FF;
    text-shadow: 0px 0px 20px #A566FF;
}
.uploaded-img {
    border-radius: 20px;
    box-shadow: 0 0 30px #7e57c2;
}
.result-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid #a066ff33;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 30px #4a148c55;
}
.progress-bar {
    height: 12px;
    border-radius: 8px;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ====================================
# LOAD MODELS
# ====================================
@st.cache_resource
def load_models():
    yolo_model = YOLO("Model/Intan Humaira_Laporan 4.pt")
    classifier = tf.keras.models.load_model("Model/Intan Humaira_Laporan2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ====================================
# HEADER
# ====================================
st.markdown("<h1>🤖 NEURA VISION</h1>", unsafe_allow_html=True)
st.markdown("<h3>AI Futuristic Dashboard by Intan Humaira</h3>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #7e57c2;'>", unsafe_allow_html=True)

# ====================================
# SIDEBAR
# ====================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=120)
    st.markdown("## 🔮 Pilihan Mode")
    mode = st.radio("Pilih Mode:", ["🧩 Deteksi Objek (YOLO)", "🧠 Klasifikasi Gambar"])
    st.markdown("---")
    uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])
    st.caption("Format file: JPG, JPEG, atau PNG")

# ====================================
# MAIN SECTION
# ====================================
col1, col2 = st.columns([0.55, 0.45])

with col1:
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True, output_format="JPEG", className="uploaded-img")

        with st.spinner("🧠 NeuraVision sedang menganalisis gambar..."):
            time.sleep(2.5)  # Simulasi proses AI
    else:
        st.info("⬅️ Silakan unggah gambar di sidebar untuk memulai analisis.")

with col2:
    if uploaded_file:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        if mode == "🧩 Deteksi Objek (YOLO)":
            st.markdown("### 🔍 Hasil Deteksi Objek")
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, use_container_width=True)
            st.success("✅ Objek berhasil terdeteksi!")

        elif mode == "🧠 Klasifikasi Gambar":
            st.markdown("### 🧬 Hasil Klasifikasi AI")
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = float(np.max(prediction))

            progress_color = "#4CAF50" if confidence > 0.7 else "#FF9800"
            st.markdown(f"<h2 style='color:#fff;'>Kelas: <b>{class_index}</b></h2>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#ccc;'>Probabilitas: {confidence*100:.2f}%</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:{progress_color}; width:{confidence*100}%; height:12px; border-radius:8px;"></div>
            """, unsafe_allow_html=True)

            quotes = [
                "“AI never sleeps, it just keeps learning.”",
                "“Neural networks see what eyes can’t.”",
                "“Confidence defines intelligence.”"
            ]
            st.markdown(f"<i>{random.choice(quotes)}</i>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ====================================
# FOOTER
# ====================================
st.markdown("<hr style='border:1px solid #7e57c2;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#888;'>© 2025 NEURA VISION | Crafted by <b>Intan Humaira</b> 🪄</p>",
    unsafe_allow_html=True
)
