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
    box-shadow: 0 0 25px rgba(0,150,255,0.2
