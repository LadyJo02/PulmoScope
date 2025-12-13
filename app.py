import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.preprocess import load_audio, create_mel
from utils.inference import load_model, predict
from utils.gradcam import GradCAM

# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="PulmoScope",
    page_icon="🫁",
    layout="wide",
)

# Background styling (simple clinical light-gray background)
st.markdown("""
<style>
body {
    background-color: #F6F7F9;
}
.section-card {
    background: white;
    padding: 18px 22px;
    border-radius: 10px;
    border: 1px solid #E5E7EB;
    margin-bottom: 22px;
}
h2, h3 {
    color: #374151;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

LABELS = ["COPD", "Healthy", "Pneumonia", "Other"]

# Session defaults
if "mel" not in st.session_state:
    st.session_state.mel = None
if "audio" not in st.session_state:
    st.session_state.audio = None

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div style='text-align:center; margin-bottom: 18px;'>
    <h1 style='margin-bottom: 4px;'>PulmoScope</h1>
    <p style='color:#6B7280; font-size:16px; margin-top:-5px;'>
        AI-assisted analysis of lung auscultation recordings
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    pure = load_model("models/pure_tcn_weights.pth", "tcn")
    snn = load_model("models/tcn_snn_weights.pth", "snn")
    return pure, snn

pure_tcn, tcn_snn = load_models()

with st.container():
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Model Status")
    c1, c2 = st.columns(2)
    c1.info(f"Pure TCN Model: {'Loaded' if pure_tcn else 'Failed'}")
    c2.info(f"Hybrid TCN-SNN Model: {'Loaded' if tcn_snn else 'Failed'}")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# AUDIO INPUT SECTION
# =========================================================
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("1. Audio Input")

input_method = st.radio(
    "Select input method",
    ["Upload WAV file", "Record using microphone"],
    horizontal=True
)

audio = None

# Upload
if input_method == "Upload WAV file":
    f = st.file_uploader("Upload lung sound (.wav)", type=["wav"])
    if f is not None:
        st.audio(f)
        audio = load_audio(f)
        st.session_state.audio = audio

# Recording
else:
    rec = st.audio_input("Record lung sound")
    if rec is not None:
        st.audio(rec)
        audio = load_audio(rec)
        st.session_state.audio = audio

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PROCESSING TRIGGER
# =========================================================
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("2. Preprocessing & Feature Extraction")

if st.session_state.audio is not None:
    if st.button("Generate Spectrogram", type="primary"):
        with st.spinner("Processing recording..."):
            st.session_state.mel = create_mel(st.session_state.audio)
else:
    st.info("Upload or record a lung sound to begin.")

st.markdown("</div>", unsafe_allow_html=True)

mel = st.session_state.mel

# =========================================================
# SPECTROGRAM VISUALIZATION
# =========================================================
if mel is not None:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Mel-Spectrogram")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    st.pyplot(fig, clear_figure=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# MODEL INFERENCE
# =========================================================
pred_tcn = pred_snn = None
prob_tcn = prob_snn = None

if mel is not None:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("3. AI Model Predictions")

    col_tcn, col_snn = st.columns(2)

    pred_tcn, prob_tcn = predict(pure_tcn, mel)
    pred_snn, prob_snn = predict(tcn_snn, mel)

    # ----------- TCN Column -----------
    with col_tcn:
        st.markdown("### Pure TCN")
        st.write(f"Predicted Class: **{pred_tcn}**")

        for label, p in zip(LABELS, prob_tcn):
            st.write(f"{label}: {p*100:.1f}%")
            st.progress(float(p))

    # ----------- TCN-SNN Column -----------
    with col_snn:
        st.markdown("### Hybrid TCN-SNN")
        st.write(f"Predicted Class: **{pred_snn}**")

        for label, p in zip(LABELS, prob_snn):
            st.write(f"{label}: {p*100:.1f}%")
            st.progress(float(p))

    # Agreement logic
    if pred_tcn == pred_snn:
        st.success(f"Both models agree on the diagnosis: {pred_tcn}")
    else:
        st.warning("Models produce different predictions. Review interpretability heatmaps below.")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# GRAD-CAM INTERPRETATION
# =========================================================
if mel is not None and pred_tcn is not None:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("4. Interpretability (Attention Heatmaps)")

    try:
        x = torch.tensor(mel).unsqueeze(0).float()

        cam_tcn = GradCAM(pure_tcn)
        cam_snn = GradCAM(tcn_snn)

        cls_idx_tcn = LABELS.index(pred_tcn)
        cls_idx_snn = LABELS.index(pred_snn)

        heat_tcn = cam_tcn.generate(x, cls_idx_tcn)
        heat_snn = cam_snn.generate(x, cls_idx_snn)

        c1, c2, c3 = st.columns(3)

        # Original Spectrogram
        with c1:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(mel, aspect="auto", cmap="viridis", origin="lower")
            ax.set_title("Mel-Spectrogram")
            st.pyplot(fig, clear_figure=True)

        # TCN Attention
        with c2:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(heat_tcn, aspect="auto", cmap="inferno", origin="lower")
            ax.set_title("Pure TCN Attention")
            st.pyplot(fig, clear_figure=True)

        # SNN Attention
        with c3:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(heat_snn, aspect="auto", cmap="inferno", origin="lower")
            ax.set_title("Hybrid TCN-SNN Attention")
            st.pyplot(fig, clear_figure=True)

    except:
        st.error("Interpretability unavailable for this sample.")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# RESET BUTTON
# =========================================================
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
if st.button("Analyze another recording"):
    st.session_state.clear()
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style='text-align:center; margin-top:28px; color:#9CA3AF; font-size:14px;'>
PulmoScope © 2025<br>Academic prototype — not a medical device.
</div>
""", unsafe_allow_html=True)
