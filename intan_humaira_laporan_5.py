import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import numpy as np
import requests, io, os
import plotly.graph_objects as go
from streamlit_lottie import st_lottie

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Neura HoloLab 3D — USK Statistics", page_icon="🛸", layout="wide")

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

def image_to_base64_str(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_models():
    yolo_path = "Model/Intan Humaira_Laporan 4.pt"
    keras_path = "Model/Intan Humaira_Laporan2.h5"

    yolo_model = YOLO(yolo_path) if os.path.exists(yolo_path) else None
    classifier = tf.keras.models.load_model(keras_path) if os.path.exists(keras_path) else None
    return yolo_model, classifier

yolo_model, classifier = load_models()

# -------------------------
# STYLING (CSS)
# -------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 10% 10%, #081229 0%, #010312 100%);
    color: #dff4ff;
    font-family: 'Poppins', sans-serif;
}
.header-container {
    display: flex;
    align-items: center;
    gap: 18px;
    background: linear-gradient(90deg, rgba(20,40,90,0.3), rgba(5,10,30,0.2));
    padding: 12px 18px;
    border-radius: 16px;
    border: 1px solid rgba(120,180,255,0.1);
    box-shadow: 0 4px 25px rgba(20,100,255,0.2);
    margin-bottom: 25px;
}
.usk-logo {
    width: 90px;
    height: auto;
    border-radius: 12px;
    box-shadow: 0 0 20px rgba(80,160,255,0.25);
}
.title-text {
    font-size: 22px;
    font-weight: 700;
    color: #9fd7ff;
}
.subtitle {
    color: #a6c9ff;
    font-size: 15px;
}
.floating-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
    border-radius: 14px;
    padding: 16px;
    border: 1px solid rgba(120,180,255,0.1);
    box-shadow: 0 4px 20px rgba(20,80,180,0.4);
    backdrop-filter: blur(6px);
    margin-bottom: 20px;
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

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.markdown("### 🧠 Mode Analisis")
    mode = st.radio("", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])
    show_chart = st.checkbox("Tampilkan Grafik", value=True)
    show_lottie = st.checkbox("Tampilkan Animasi", value=True)

# -------------------------
# LOTTIE (optional)
# -------------------------
if show_lottie:
    holo_anim = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json")
    if holo_anim:
        st_lottie(holo_anim, height=180, key="anim")

# -------------------------
# MAIN CONTENT
# -------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    if mode == "Deteksi Objek (YOLO)":
        with st.spinner("🔍 Sedang mendeteksi objek..."):
            results = yolo_model(img)
        plotted = results[0].plot()
        if isinstance(plotted, np.ndarray):
            plotted = Image.fromarray(plotted)
        st.image(plotted, caption="Hasil Deteksi Objek", use_container_width=True)

        # Hitung objek
        obj_counts = {}
        for cls in results[0].boxes.cls:
            label = results[0].names[int(cls)]
            obj_counts[label] = obj_counts.get(label, 0) + 1

        if show_chart:
            fig = go.Figure(go.Bar(
                x=list(obj_counts.keys()),
                y=list(obj_counts.values()),
                marker_color="#3da9fc",
                text=list(obj_counts.values()),
                textposition="outside"
            ))
            fig.update_layout(
                title="📊 Jumlah Objek Terdeteksi",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#dff4ff"),
                yaxis=dict(gridcolor="rgba(100,150,255,0.1)")
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        if classifier is None:
            st.error("Model klasifikasi tidak ditemukan.")
        else:
            arr = np.expand_dims(image.img_to_array(img.resize((224, 224))) / 255.0, axis=0)
            pred = classifier.predict(arr)
            labels = ["freshapples","freshbanana","freshoranges","rottenapples","rottenbanana","rottenoranges"]
            fig = go.Figure(go.Bar(
                x=labels, y=pred[0],
                marker_color="#9fd7ff",
                text=[f"{p*100:.1f}%" for p in pred[0]],
                textposition="outside"
            ))
            fig.update_layout(
                title="📈 Probabilitas Kelas Gambar",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#dff4ff")
            )
            st.plotly_chart(fig, use_container_width=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#9fd7ff;font-size:13px;'>© Universitas Syiah Kuala — Neura HoloLab 3D | Crafted by Intan Humaira</div>", unsafe_allow_html=True)
