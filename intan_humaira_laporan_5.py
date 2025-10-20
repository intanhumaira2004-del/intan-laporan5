import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import requests, random, time
from streamlit_lottie import st_lottie
import base64
from io import BytesIO

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="Neura HoloUltimate", page_icon="🛸", layout="wide")

# ==========================
# FUNSI LOTTIE
# ==========================
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_scan = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_gs1k4t7p.json")

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
# SIDEBAR
# ==========================
with st.sidebar:
    st.title("🛸 Neura HoloUltimate")
    mode = st.radio("Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
    st.caption("Upload file JPG/PNG untuk memulai holographic AI analysis")

# ==========================
# CUSTOM CSS FUTURISTIK
# ==========================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at 30% 30%, #0d0d0d, #1a0033);
    color: #eee;
    font-family: 'Poppins', sans-serif;
}
h1,h2,h3 {
    color:#00FFF7;
    text-align:center;
    text-shadow:0 0 20px #00FFF7;
}
.floating-card {
    background: rgba(0,255,255,0.05);
    border: 2px solid #00FFF7;
    border-radius:20px;
    padding:20px;
    backdrop-filter:blur(15px);
    box-shadow:0 0 40px #00FFF7;
    transition: transform 0.3s;
}
.floating-card:hover {
    transform: translateY(-10px) scale(1.05);
}
.circular-gauge {
    width:120px;
    height:120px;
    border-radius:50%;
    background: conic-gradient(#00FFF7 0%, #0d0d0d 0%);
    display:flex;
    justify-content:center;
    align-items:center;
    color:#00FFF7;
    font-weight:bold;
    font-size:18px;
    text-shadow:0 0 8px #00FFF7;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown("<h1>🛸 NEURA HOLOULTIMATE</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #00FFF7;'>", unsafe_allow_html=True)

# ==========================
# MAIN CONTENT
# ==========================
def image_to_base64(img: Image.Image):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

if uploaded_file:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns([0.5,0.5])

    with col1:
        # Tampilkan gambar dengan efek rounded + shadow via HTML
        img_base64 = image_to_base64(img)
        st.markdown(f"""
        <img src="data:image/png;base64,{img_base64}" style="border-radius:20px; box-shadow:0 0 30px #7e57c2; width:100%;">
        """, unsafe_allow_html=True)
        st_lottie(lottie_scan, height=300, key="holo_scan")

    with col2:
        st.markdown("<div class='floating-card'>", unsafe_allow_html=True)

        if mode == "Deteksi Objek (YOLO)":
            st.markdown("<h3>🔍 Hasil Deteksi Objek</h3>", unsafe_allow_html=True)
            with st.spinner("Memindai objek holografik..."):
                time.sleep(2)
                results = yolo_model(img)
                result_img = results[0].plot()
                result_base64 = image_to_base64(result_img)
                st.markdown(f"""
                <img src="data:image/png;base64,{result_base64}" style="border-radius:20px; width:100%; box-shadow:0 0 20px #00FFF7;">
                """, unsafe_allow_html=True)
            st.success("✅ Objek berhasil terdeteksi!")
            st.markdown("<p style='color:#00FFF7;'>💡 Klik objek di gambar untuk info (simulasi tooltip interaktif)</p>", unsafe_allow_html=True)

        elif mode == "Klasifikasi Gambar":
            st.markdown("<h3>🧬 Hasil Klasifikasi AI</h3>", unsafe_allow_html=True)
            img_resized = img.resize((224,224))
            img_array = np.expand_dims(image.img_to_array(img_resized)/255.0, axis=0)

            with st.spinner("AI sedang memproses prediksi..."):
                time.sleep(2)
                pred = classifier.predict(img_array)
                class_idx = np.argmax(pred)
                conf = float(np.max(pred))

            # Circular holographic gauge
            st.markdown(f"""
            <div class="circular-gauge" style="background: conic-gradient(#00FFF7 {conf*100}%, #0d0d0d 0%);">
            {conf*100:.1f}%
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#00FFF7; text-shadow:0 0 15px #00FFF7;'>Kelas: {class_idx}</h2>", unsafe_allow_html=True)

            quotes = ["“AI sees beyond human eyes.”","“Neural networks never sleep.”","“Confidence defines intelligence.”"]
            st.markdown(f"<i style='color:#00fff7;'>{random.choice(quotes)}</i>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("⬅️ Unggah gambar untuk memulai analisis holographic AI.")

# ==========================
# FOOTER
# ==========================
st.markdown("<hr style='border:1px solid #00FFF7;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#00FFFF; text-shadow:0 0 10px #00FFFF;'>© 2025 NEURA HOLOULTIMATE | Crafted by Intan Humaira</p>", unsafe_allow_html=True)
