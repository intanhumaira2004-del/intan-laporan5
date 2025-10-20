import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import requests, random
import plotly.graph_objects as go
from io import BytesIO
import base64
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Neura HoloLab 3D", page_icon="🛸", layout="wide")

def load_lottie_url(url):
try:
r = requests.get(url)
if r.status_code == 200:
return r.json()
else:
return None
except:
return None

lottie_scan = load_lottie_url("https://lottie.host/30b58e1c-6b67-4b8a-bbb8-bb8ee1b26b47/1abGxnmqlq.json
")
if lottie_scan is None:
lottie_scan = {"v": "5.7.4","fr": 30,"ip": 0,"op": 60,"w": 100,"h": 100,"nm": "Fallback animation","ddd": 0,"assets": [],"layers": []}

@st.cache_resource
def load_models():
yolo_model = YOLO("Model/Intan Humaira_Laporan 4.pt")
classifier = tf.keras.models.load_model("Model/Intan Humaira_Laporan2.h5")
return yolo_model, classifier

yolo_model, classifier = load_models()

with st.sidebar:
st.title("🛸 Neura HoloLab 3D")
mode = st.radio("Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
st.caption("Upload file untuk analisis AI holographic")

st.markdown("""

<style> body { background: linear-gradient(145deg, #0a0f24, #1b2a49); color: #f2f2f2; font-family: 'Poppins', sans-serif;} h1,h2,h3 { color:#4da6ff; text-align:center; text-shadow:0 0 20px #1a75ff;} hr {border:1px solid #4da6ff;} .floating-card { background: rgba(30,60,120,0.15); border: 2px solid #4da6ff; border-radius:20px; padding:20px; backdrop-filter:blur(15px); box-shadow:0 0 30px #0047b3; transition: transform 0.3s; } .floating-card:hover { transform: translateY(-10px) scale(1.05);} .circular-gauge { width:120px; height:120px; border-radius:50%; display:flex; justify-content:center; align-items:center; color:#4da6ff; font-weight:bold; font-size:18px; text-shadow:0 0 8px #4da6ff; } </style>

""", unsafe_allow_html=True)

st.markdown("<h1>🛸 NEURA HOLOLAB 3D</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

def image_to_base64(img: Image.Image):
buffered = BytesIO()
img.save(buffered, format="PNG")
return base64.b64encode(buffered.getvalue()).decode()

def display_image(img, rotate=False):
img_base64 = image_to_base64(img)
transform = "rotateY(15deg)" if rotate else "none"
st.markdown(f"""
<img src="data:image/png;base64,{img_base64}" style="border-radius:20px; box-shadow:0 0 40px #0047b3; width:100%; transform:{transform};">
""", unsafe_allow_html=True)

def run_yolo(img):
results = yolo_model(img)
result_img = results[0].plot()
display_image(Image.fromarray(result_img))

obj_counts = {}
for box in results[0].boxes.cls:
    label = results[0].names[int(box)]
    obj_counts[label] = obj_counts.get(label, 0) + 1

if obj_counts:
    fig = go.Figure(go.Bar(
        x=list(obj_counts.keys()),
        y=list(obj_counts.values()),
        marker_color='#4da6ff'
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#4da6ff',
        title="📊 Jumlah Objek Terdeteksi"
    )
    st.plotly_chart(fig, use_container_width=True)


def run_classifier(img):
img_resized = img.resize((224, 224))
img_array = np.expand_dims(image.img_to_array(img_resized) / 255.0, axis=0)
pred = classifier.predict(img_array)
class_idx = np.argmax(pred)
conf = float(np.max(pred))

labels = ["Fresh Apple", "Fresh Banana", "Fresh Orange", "Rotten Apple", "Rotten Banana", "Rotten Orange"]
fig = go.Figure(go.Bar(x=labels, y=pred[0], marker_color='#4da6ff'))
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='#4da6ff',
    title="📊 Probabilitas Kelas"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
    <div class="circular-gauge" style="background: conic-gradient(#4da6ff {conf*100}%, #0a0f24 0%);">{conf*100:.1f}%</div>
""", unsafe_allow_html=True)
st.markdown(f"<h2 style='color:#4da6ff; text-shadow:0 0 15px #4da6ff;'>Kelas Utama: {labels[class_idx]}</h2>", unsafe_allow_html=True)

quotes = [
    "“AI sees beyond human eyes.”",
    "“Neural networks never sleep.”",
    "“Confidence defines intelligence.”"
]
st.markdown(f"<i style='color:#4da6ff;'>{random.choice(quotes)}</i>", unsafe_allow_html=True)


mode_map = {"Deteksi Objek (YOLO)": run_yolo, "Klasifikasi Gambar": run_classifier}

if uploaded_file:
img = Image.open(uploaded_file)
col1, col2 = st.columns([0.5, 0.5])
with col1:
display_image(img, rotate=True)
st_lottie(lottie_scan, height=300, key="holo_scan")
with col2:
st.markdown("<div class='floating-card'>", unsafe_allow_html=True)
mode_mapmode

st.markdown("</div>", unsafe_allow_html=True)
else:
st.info("⬅️ Unggah gambar untuk memulai analisis holographic AI.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#4da6ff; text-shadow:0 0 10px #0047b3;'>© 2025 NEURA HOLOLAB 3D | Crafted by Intan Humaira</p>", unsafe_allow_html=True)
