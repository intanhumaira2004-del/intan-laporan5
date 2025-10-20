import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import requests, random, time
from streamlit_lottie import st_lottie
import plotly.graph_objects as go
from io import BytesIO
import base64

# ==========================
# Config halaman
# ==========================
st.set_page_config(page_title="Neura HoloLab 3D", page_icon="🛸", layout="wide")

# ==========================
# Load Lottie
# ==========================
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

lottie_scan = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_gs1k4t7p.json")

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("Model/Intan Humaira_Laporan 4.pt")
    classifier = tf.keras.models.load_model("Model/Intan Humaira_Laporan2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Sidebar
# ==========================
with st.sidebar:
    st.title("🛸 Neura HoloLab 3D")
    mode = st.radio("Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg","jpeg","png"])
    st.caption("Upload file untuk analisis AI holographic")

# ==========================
# CSS Futuristik
# ==========================
st.markdown("""
<style>
body { background: radial-gradient(circle at 30% 30%, #0d0d0d, #1a0033); color: #eee; font-family: 'Poppins', sans-serif;}
h1,h2,h3 { color:#00FFF7; text-align:center; text-shadow:0 0 20px #00FFF7;}
.floating-card {background: rgba(0,255,255,0.05); border: 2px solid #00FFF7; border-radius:20px;
                padding:20px; backdrop-filter:blur(15px); box-shadow:0 0 40px #00FFF7; transition: transform 0.3s;}
.floating-card:hover { transform: translateY(-10px) rotateX(5deg) scale(1.05);}
.circular-gauge {width:120px; height:120px; border-radius:50%; display:flex; justify-content:center; align-items:center;
                 color:#00FFF7; font-weight:bold; font-size:18px; text-shadow:0 0 8px #00FFF7;}
</style>
""", unsafe_allow_html=True)

# ==========================
# Header
# ==========================
st.markdown("<h1>🛸 NEURA HOLOLAB 3D</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #00FFF7;'>", unsafe_allow_html=True)

# ==========================
# Fungsi bantu
# ==========================
def image_to_base64(img: Image.Image):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def display_image(img, rotate=False):
    img_base64 = image_to_base64(img)
    transform = "rotateY(15deg)" if rotate else "none"
    st.markdown(f"""
        <img src="data:image/png;base64,{img_base64}" style="border-radius:20px; box-shadow:0 0 40px #7e57c2; width:100%; transform:{transform};">
    """, unsafe_allow_html=True)

# ==========================
# Fungsi Mode Mapping
# ==========================
def run_yolo(img):
    results = yolo_model(img)
    result_img = results[0].plot()
    display_image(result_img)
    
    # Pie chart objek
    obj_counts = {}
    for box in results[0].boxes.cls:
        label = results[0].names[int(box)]
        obj_counts[label] = obj_counts.get(label,0)+1
    if obj_counts:
        fig = go.Figure(go.Pie(labels=list(obj_counts.keys()), values=list(obj_counts.values()),
                               hole=0.4, marker_colors=['#00FFF7','#0ff','#0aa','#0f5']))
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font_color='cyan', title="📊 Komposisi Objek")
        st.plotly_chart(fig, use_container_width=True)

def run_classifier(img):
    img_resized = img.resize((224,224))
    img_array = np.expand_dims(image.img_to_array(img_resized)/255.0, axis=0)
    pred = classifier.predict(img_array)
    class_idx = np.argmax(pred)
    conf = float(np.max(pred))
    
    # Bar chart neon
    labels = [f"Kelas {i}" for i in range(len(pred[0]))]
    fig = go.Figure(go.Bar(x=labels, y=pred[0], marker_color='cyan'))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font_color='cyan', title="📊 Probabilitas Kelas")
    st.plotly_chart(fig, use_container_width=True)
    
    # Circular gauge
    st.markdown(f"""
        <div class="circular-gauge" style="background: conic-gradient(#00FFF7 {conf*100}%, #0d0d0d 0%);">{conf*100:.1f}%</div>
        """, unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:#00FFF7; text-shadow:0 0 15px #00FFF7;'>Kelas Utama: {class_idx}</h2>", unsafe_allow_html=True)
    
    # AI Quotes
    quotes = ["“AI sees beyond human eyes.”","“Neural networks never sleep.”","“Confidence defines intelligence.”"]
    st.markdown(f"<i style='color:#00fff7;'>{random.choice(quotes)}</i>", unsafe_allow_html=True)

# ==========================
# Jalankan mode sesuai mapping
# ==========================
mode_map = {
    "Deteksi Objek (YOLO)": run_yolo,
    "Klasifikasi Gambar": run_classifier
}

if uploaded_file:
    img = Image.open(uploaded_file)
    col1,col2 = st.columns([0.5,0.5])
    with col1:
        display_image(img, rotate=True)
        st_lottie(lottie_scan, height=300, key="holo_scan")
    with col2:
        st.markdown("<div class='floating-card'>", unsafe_allow_html=True)
        mode_map[mode](img)  # Panggil fungsi sesuai mode tanpa if-else
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("⬅️ Unggah gambar untuk memulai analisis holographic AI.")

# ==========================
# Footer
# ==========================
st.markdown("<hr style='border:1px solid #00FFF7;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#00FFFF; text-shadow:0 0 10px #00FFFF;'>© 2025 NEURA HOLOLAB 3D | Crafted by Intan Humaira</p>", unsafe_allow_html=True)
