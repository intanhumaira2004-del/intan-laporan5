import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import numpy as np
import requests, random, io, os
import plotly.graph_objects as go
from io import BytesIO
import base64
from streamlit_lottie import st_lottie

# -------------------------
# Minimal config / page
# -------------------------
st.set_page_config(page_title="Neura HoloLab 3D — USK Statistics", page_icon="🛸", layout="wide")

# -------------------------
# Helper: load Lottie safely
# -------------------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# gentle holo lottie (fallback safe)
lottie_scan = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_gs1k4t7p.json")
if lottie_scan is None:
    # very small harmless fallback Lottie-like dict so st_lottie won't crash
    lottie_scan = {"v":"5.5.7","fr":30,"ip":0,"op":60,"w":100,"h":100,"nm":"fallback","ddd":0,"assets":[],"layers":[]}

# -------------------------
# Load models (cached)
# -------------------------
@st.cache_resource
def load_models():
    # paths expected in repo:
    # Model/Intan Humaira_Laporan 4.pt  (YOLO weights)
    # Model/Intan Humaira_Laporan2.h5   (Keras classifier) - optional
    yolo_path = "Model/Intan Humaira_Laporan 4.pt"
    keras_path = "Model/Intan Humaira_Laporan2.h5"

    yolo_model = None
    classifier = None

    # load YOLO (Ultralytics)
    try:
        if os.path.exists(yolo_path):
            yolo_model = YOLO(yolo_path)
        else:
            # Try loading from root if placed differently
            yolo_model = YOLO(yolo_path)
    except Exception as e:
        yolo_model = None
        st.error("⚠️ Gagal memuat YOLO model. Pastikan file 'Model/Intan Humaira_Laporan 4.pt' ada di repo. Error: " + str(e))

    # load Keras classifier (optional)
    try:
        if os.path.exists(keras_path):
            classifier = tf.keras.models.load_model(keras_path)
        else:
            classifier = None
    except Exception as e:
        classifier = None
        # do not crash — classifier optional
        st.warning("⚠️ Gagal memuat Keras classifier (.h5). Klasifikasi akan non-aktif. Error: " + str(e))

    return yolo_model, classifier

yolo_model, classifier = load_models()

# -------------------------
# Styling: holographic + logo animation + particles
# -------------------------
st.markdown(
    """
    <style>
    /* overall background */
    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(6,24,58,0.9), rgba(3,8,29,0.95) 30%, #030416 100%);
        color: #dff4ff;
    }

    /* floating header card */
    .header-card {
        display:flex;
        align-items:center;
        gap:18px;
        padding:14px 18px;
        border-radius:14px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(77,166,255,0.12);
        box-shadow: 0 8px 30px rgba(10,40,90,0.5);
        backdrop-filter: blur(6px);
    }

    .site-title {
        font-family: 'Poppins', sans-serif;
        font-size:22px;
        font-weight:700;
        color:#9fd7ff;
        text-shadow: 0 0 18px rgba(77,166,255,0.12);
    }

    /* logo rotate */
    .usk-logo {
        width:92px;
        height: auto;
        border-radius:10px;
        padding:6px;
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.03);
        transform-origin: center;
        animation: slow-rotate 12s linear infinite;
        box-shadow: 0 6px 26px rgba(0,80,160,0.25);
    }

    @keyframes slow-rotate {
        0% { transform: rotateZ(0deg) translateZ(0); }
        50% { transform: rotateZ(6deg) translateZ(0); }
        100% { transform: rotateZ(0deg) translateZ(0); }
    }

    /* floating card for content */
    .floating-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 14px;
        padding: 16px;
        border: 1px solid rgba(77,166,255,0.08);
        box-shadow: 0 10px 40px rgba(3,8,29,0.6);
        backdrop-filter: blur(8px);
    }

    /* holo particle overlay - subtle */
    .particle-layer {
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            radial-gradient(rgba(77,166,255,0.04) 1px, transparent 1px),
            radial-gradient(rgba(77,166,255,0.03) 1px, transparent 1px);
        background-size: 120px 120px, 64px 64px;
        opacity: 0.45;
        mix-blend-mode: screen;
        transform: translateZ(0);
        animation: particle-move 40s linear infinite;
    }
    @keyframes particle-move {
        from { background-position: 0 0, 0 0; }
        to   { background-position: -2400px 1200px, 1200px -800px; }
    }

    /* small helpers */
    .muted { color: rgba(160,200,230,0.7); font-size:14px; }
    </style>
    """,
    unsafe_allow_html=True
)

# particle overlay (pure-css)
st.markdown('<div class="particle-layer"></div>', unsafe_allow_html=True)

# -------------------------
# Header with USK logo (left) + title
# -------------------------
# We expect logo file placed at "assets/usk_logo.png" inside the repo.
logo_path_candidates = [
    "assets/usk_logo.png",
    "assets/usk_logo.jpg",
    "assets/usk_logo.svg",
    "usk_logo.png",
    "usk_logo.jpg"
]

logo_bytes = None
logo_src = None
for p in logo_path_candidates:
    if os.path.exists(p):
        logo_path = p
        try:
            with open(logo_path, "rb") as f:
                logo_bytes = f.read()
                logo_src = "local"
                break
        except Exception:
            logo_bytes = None
            logo_src = None

# header layout
col_h1, col_h2 = st.columns([0.18, 0.82])
with col_h1:
    if logo_bytes is not None:
        st.image(logo_bytes, width=92, use_column_width=False, output_format="PNG")
    else:
        # fallback: small stylized text block
        st.markdown("<div style='width:92px;height:92px;border-radius:10px;padding:10px;background:linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01));display:flex;align-items:center;justify-content:center;border:1px solid rgba(255,255,255,0.03);'><span style='color:#cfeeff;font-weight:700'>USK</span></div>", unsafe_allow_html=True)

with col_h2:
    st.markdown("<div class='header-card'><div><div class='site-title'>Neura HoloLab 3D — USK Statistics</div><div class='muted'>Faculty of Mathematics and Natural Sciences</div></div></div>", unsafe_allow_html=True)

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio("", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    st.markdown("**Pengaturan**")
    show_legend = st.checkbox("Tampilkan legend pada grafik", value=False)
    show_lottie = st.checkbox("Tampilkan animasi Holo", value=True)

# lottie
if show_lottie:
    try:
        st_lottie(lottie_scan, height=140, key="small_holo", quality="low")
    except Exception:
        pass

# -------------------------
# Utility: image -> base64 (for inline)
# -------------------------
def image_to_base64_str(img: Image.Image):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# -------------------------
# Core functions: YOLO detect + bar chart (stylish)
# -------------------------
def render_detection_and_chart(pil_img: Image.Image):
    if yolo_model is None:
        st.error("YOLO model belum terload. Pastikan file 'Model/Intan Humaira_Laporan 4.pt' ada di repo.")
        return

    with st.spinner("🔍 Mendeteksi objek..."):
        results = yolo_model(pil_img)

    # overlay boxes: use results[0].plot()
    try:
        plotted = results[0].plot()
        # results[0].plot() returns numpy array or PIL; ensure PIL
        if isinstance(plotted, np.ndarray):
            plotted_img = Image.fromarray(plotted)
        else:
            plotted_img = plotted
    except Exception:
        plotted_img = pil_img

    st.image(plotted_img, caption="Hasil Deteksi (bounding box)", use_column_width=True)

    # count classes
    obj_counts = {}
    try:
        for cls in results[0].boxes.cls:
            label = results[0].names[int(cls)]
            obj_counts[label] = obj_counts.get(label, 0) + 1
    except Exception:
        # fallback using pandas output
        try:
            df = results[0].pandas().xyxy[0]
            names = df['name'].tolist()
            for n in names:
                obj_counts[n] = obj_counts.get(n, 0) + 1
        except Exception:
            pass

    # if no detected
    if not obj_counts:
        st.warning("🚫 Tidak ada objek terdeteksi dalam gambar ini.")
        return

    # create styled plotly bar chart
    labels = list(obj_counts.keys())
    values = [obj_counts[k] for k in labels]

    # color palette - blue dark variations
    base_colors = [
        "#7fbfff", "#66b2ff", "#4da6ff", "#3388ff", "#1966ff", "#0a3fff"
    ]
    bar_colors = [base_colors[i % len(base_colors)] for i in range(len(labels))]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.06)", width=1.5)),
            text=[str(v) for v in values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Jumlah: %{y}<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(text="📊 Jumlah Objek Terdeteksi", x=0.5, font=dict(color="#bfe7ff", size=20)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#bfe7ff"),
        xaxis=dict(showgrid=False, tickangle=0),
        yaxis=dict(showgrid=True, gridcolor='rgba(77,166,255,0.08)'),
        margin=dict(l=40, r=20, t=80, b=40),
        showlegend=show_legend
    )

    st.plotly_chart(fig, use_container_width=True)

    # simple summary (fresh vs rotten if naming consistent)
    fresh_total = sum(v for k, v in obj_counts.items() if "fresh" in k.lower())
    rotten_total = sum(v for k, v in obj_counts.items() if "rotten" in k.lower())
    st.markdown(f"<div class='floating-card' style='text-align:center'><strong style='color:#bfe7ff'>Deteksi ringkasan:</strong> &nbsp; <span style='color:#9fd7ff'>{fresh_total}</span> buah segar — &nbsp; <span style='color:#ffb379'>{rotten_total}</span> buah busuk</div>", unsafe_allow_html=True)


# -------------------------
# Core functions: Keras classifier view (if present)
# -------------------------
def render_classifier(pil_img: Image.Image):
    if classifier is None:
        st.error("Classifier (Intan Humaira_Laporan2.h5) tidak tersedia. Pastikan file ada di folder Model/ bila ingin menggunakan mode klasifikasi.")
        return

    img_resized = pil_img.resize((224, 224)).convert("RGB")
    arr = image.img_to_array(img_resized) / 255.0
    arr = np.expand_dims(arr, 0)
    pred = classifier.predict(arr)
    pred = np.squeeze(pred)
    class_idx = int(np.argmax(pred))
    conf = float(np.max(pred))

    labels = ["freshapples","freshbanana","freshoranges","rottenapples","rottenbanana","rottenoranges"]
    # ensure length match
    if len(pred) != len(labels):
        # fallback label creation
        labels = [f"Kelas {i}" for i in range(len(pred))]

    # plot probabilities as a neon bar
    fig = go.Figure(go.Bar(
        x=labels,
        y=pred,
        marker=dict(color=["#66b2ff" for _ in labels]),
        text=[f"{p*100:.1f}%" for p in pred],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Prob: %{y:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="📊 Probabilitas Kelas (Klasifier)", x=0.5, font=dict(color="#bfe7ff")),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#bfe7ff"),
        margin=dict(l=30, r=30, t=60, b=40),
        yaxis=dict(showgrid=True, gridcolor='rgba(77,166,255,0.06)')
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<div style='text-align:center;color:#bfe7ff'>Kelas utama: <strong style='color:#9fd7ff'>{labels[class_idx]}</strong> — Confidence: <strong>{conf*100:.1f}%</strong></div>", unsafe_allow_html=True)


# -------------------------
# Main: use uploaded file; if none, show sample placeholders
# -------------------------
if uploaded_file is None:
    # show sample gallery (if you want to include sample images in repo, put them under 'sample_image/')
    st.info("Silakan unggah gambar buah (jpg/png). Atau gunakan sample dari folder `sample_image/` jika tersedia.")
    sample_cols = st.columns(3)
    sample_paths = []
    for root, dirs, files in os.walk("sample_image"):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                sample_paths.append(os.path.join(root, f))
    # show up to 3 sample thumbnails
    for i in range(3):
        with sample_cols[i]:
            if i < len(sample_paths):
                try:
                    thumb = Image.open(sample_paths[i])
                    thumb = ImageOps.fit(thumb, (280, 180))
                    st.image(thumb, use_column_width=True)
                    if st.button(f"Pakai contoh {i+1}"):
                        uploaded_file = open(sample_paths[i], "rb")
                        # convert to BytesIO to mimic uploaded_file behavior
                        uploaded_bytes = io.BytesIO(uploaded_file.read())
                        uploaded_file = uploaded_bytes
                        # proceed below by falling through
                except Exception:
                    st.write("---")
            else:
                st.write("---")
else:
    try:
        pil_img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error("Gagal membuka file gambar: " + str(e))
        pil_img = None

    if pil_img is not None:
        if mode == "Deteksi Objek (YOLO)":
            render_detection_and_chart(pil_img)
        else:
            # classification mode
            render_classifier(pil_img)

# -------------------------
# Footer
# -------------------------
st.markdown("<hr style='border:1px solid rgba(100,181,246,0.12)'>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#bfe7ff;font-size:13px'>© Universitas Syiah Kuala — Neura HoloLab 3D | Crafted by Intan Humaira</div>", unsafe_allow_html=True)
