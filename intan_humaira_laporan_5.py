

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("Model/Intan Humaira_Laporan 4.pt")
    classifier = tf.keras.models.load_model("Model/Intan Humaira_Laporan2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="Hologram AI Vision", page_icon="🧠", layout="wide")

# ==========================
# CUSTOM CSS FUTURISTIK
# ==========================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 20% 20%, rgba(10,20,45,0.95), rgba(2,5,18,1) 90%);
    color: #e7f4ff;
    font-family: 'Poppins', sans-serif;
}
.header-card {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 18px 22px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
    border: 1px solid rgba(77,166,255,0.15);
    box-shadow: 0 8px 40px rgba(8,70,200,0.2);
    backdrop-filter: blur(10px);
}
.site-title {
    font-size: 26px;
    font-weight: 700;
    background: linear-gradient(90deg, #8fd6ff, #70b7ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 15px rgba(77,166,255,0.3);
}
.muted {
    color: rgba(190, 220, 255, 0.7);
    font-size: 14px;
}
.usk-logo {
    width: 95px;
    height: auto;
    border-radius: 14px;
    padding: 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(120,180,255,0.15);
    box-shadow: 0 0 25px rgba(0,150,255,0.25);
    animation: rotate-slow 14s ease-in-out infinite alternate;
}
@keyframes rotate-slow {
    from { transform: rotateZ(-6deg); }
    to { transform: rotateZ(6deg); }
}
.floating-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
    border-radius: 16px;
    padding: 16px;
    border: 1px solid rgba(120,180,255,0.1);
    box-shadow: 0 10px 35px rgba(3,8,29,0.6);
    backdrop-filter: blur(8px);
}
.particle-layer {
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        radial-gradient(rgba(100,170,255,0.05) 1px, transparent 1px),
        radial-gradient(rgba(80,140,255,0.04) 1px, transparent 1px);
    background-size: 120px 120px, 60px 60px;
    opacity: 0.4;
    mix-blend-mode: screen;
    animation: move 40s linear infinite;
}
@keyframes move {
    from { background-position: 0 0, 0 0; }
    to   { background-position: -1200px 600px, 600px -400px; }
}
</style>
<div class="particle-layer"></div>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
col_h1, col_h2 = st.columns([0.18, 0.82])
with col_h1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/3e/Lambang_Universitas_Syiah_Kuala.png",
             caption=None, use_container_width=True)
with col_h2:
    st.markdown("""
        <div class="header-card">
            <div>
                <div class="site-title">HOLOGRAM AI DASHBOARD</div>
                <div class="muted">Intan Humaira — Vision Intelligence System</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================
# SIDEBAR MODE TANPA IF ELSE
# ==========================
mode = st.sidebar.radio("🧩 Pilih Mode:", ["Deteksi Objek YOLO", "Klasifikasi Gambar", "Visualisasi 3D"])

# ==========================
# UPLOAD IMAGE
# ==========================
img_file = st.file_uploader("📂 Unggah Gambar", type=["jpg", "jpeg", "png"])
if img_file:
    img = Image.open(img_file)
    st.image(img, caption="📸 Gambar Diupload", use_container_width=True)

# ==========================
# DICTIONARY MODE HANDLER TANPA IF ELSE
# ==========================
def detect_yolo():
    results = yolo_model(img)
    res_img = results[0].plot()
    st.image(res_img, caption="🔍 Hasil Deteksi YOLO", use_container_width=True)

def classify_image():
    resized = img.resize((224, 224))
    arr = image.img_to_array(resized)
    arr = np.expand_dims(arr, axis=0) / 255.0
    pred = classifier.predict(arr)
    idx = int(np.argmax(pred))
    conf = float(np.max(pred))

    st.subheader("🔎 Hasil Prediksi Klasifikasi")
    st.metric("Kelas", f"Class {idx}")
    st.metric("Probabilitas", f"{conf:.2%}")

    # --- grafik hasil prediksi
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Kelas {i}" for i in range(pred.shape[1])],
        y=pred[0],
        marker_color="rgba(90,200,255,0.8)"
    ))
    fig.update_layout(title="Distribusi Probabilitas Kelas",
                      template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

def visualisasi_3d():
    st.subheader("🌌 Visualisasi 3D Dinamis")
    x, y, z = np.random.rand(3, 100)
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color=z, colorscale='Viridis', opacity=0.8)
    )])
    fig.update_layout(scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        bgcolor='rgba(0,0,0,0)'
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# mapping fungsi berdasarkan mode
actions = {
    "Deteksi Objek YOLO": detect_yolo,
    "Klasifikasi Gambar": classify_image,
    "Visualisasi 3D": visualisasi_3d
}

if img_file or mode == "Visualisasi 3D":
    actions[mode]()
