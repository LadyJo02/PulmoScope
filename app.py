import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.preprocess import load_audio, create_mel
from utils.inference import load_model, predict
from utils.gradcam import GradCAM

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="PulmoScope",
    layout="wide",
)

# Subtle clinical background
st.markdown("""
<style>
body {
    background-color: #F5F7FA;
}
.block {
    background-color: white;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #E3E4E6;
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)

LABELS = ["COPD", "Healthy", "Pneumonia", "Other"]

# session vars
if "mel" not in st.session_state:
    st.session_state.mel = None
if "audio" not in st.session_state:
    st.session_state.audio = None

# =========================================================
# HEADER (Clean, Medical Style)
# =========================================================
st.markdown("""
<h1 style='text-align:center; margin-bottom:4px;'>PulmoScope</h1>
<p style='text-align:center; color:#6b7280; font-size:16px; margin-top:-10px;'>
Automated respiratory sound analysis using temporal deep learning models
</p>
""", unsafe_allow_html=True)

st.divider()

# =========================================================
# LOAD MODELS (Cached)
# =========================================================
@st.cache_resource
def load_models():
    pure = load_model("models/pure_tcn_weights.pth", "tcn")
    snn = load_model("models/tcn_snn_weights.pth", "snn")
    return pure, snn

with st.spinner("Loading PulmoScope models..."):
    pure_tcn, tcn_snn = load_models()

# =========================================================
# LAYOUT: LEFT (Input) | RIGHT (Preprocessing & Results)
# =========================================================
left, right = st.columns([0.45, 0.55])

# =========================================================
# LEFT PANEL — INPUT ZONE
# =========================================================
with left:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("1. Audio Input")

    input_mode = st.radio(
        "Choose method:",
        ["Upload WAV file", "Record using microphone"],
    )

    audio = None

    # Upload mode
    if input_mode == "Upload WAV file":
        file = st.file_uploader("Upload lung sound file (.wav)", type=["wav"])
        if file is not None:
            st.audio(file)
            audio = load_audio(file)

    # Real-time microphone recording
    if input_mode == "Record using microphone":
        recorded = st.audio_input("Record lung sound")
        if recorded is not None:
            st.audio(recorded)
            audio = load_audio(recorded)

    # Save audio to session
    if audio is not None:
        st.session_state.audio = audio

        # Audio diagnostics
        st.markdown("### Audio Diagnostics")
        st.write(f"Duration: {round(len(audio) / 16000, 2)} seconds")
        st.write("Sample rate: 16000 Hz (resampled)")
        st.write("Channels: Mono")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# RIGHT PANEL — PROCESSING & RESULTS
# =========================================================
with right:
    st.markdown("<div class='block'>", unsafe_allow_html=True)

    st.subheader("2. Preprocessing & Mel-Spectrogram")

    if st.session_state.audio is not None and st.button("Run PulmoScope Analysis"):
        with st.spinner("Processing audio and generating Mel-spectrogram..."):
            st.session_state.mel = create_mel(st.session_state.audio)

    mel = st.session_state.mel

    if mel is not None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title("Mel-Spectrogram")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.tick_params(labelsize=8)
        st.pyplot(fig, clear_figure=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 3. MODEL RESULTS (Side-by-Side)
# =========================================================
if st.session_state.mel is not None:

    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("3. Model Predictions")

    col1, col2 = st.columns(2)

    pred_tcn, prob_tcn = predict(pure_tcn, mel)
    pred_snn, prob_snn = predict(tcn_snn, mel)

    # TCN Results
    with col1:
        st.markdown("#### Pure TCN")
        st.write(f"Predicted: **{pred_tcn}**")
        for label, p in zip(LABELS, prob_tcn):
            st.write(f"{label}: {p*100:.1f}%")
            st.progress(float(p))

    # SNN Results
    with col2:
        st.markdown("#### Hybrid TCN–SNN")
        st.write(f"Predicted: **{pred_snn}**")
        for label, p in zip(LABELS, prob_snn):
            st.write(f"{label}: {p*100:.1f}%")
            st.progress(float(p))

    # agreement measure
    agreement = pred_tcn == pred_snn
    if agreement:
        st.info("Both models agreed on the same diagnosis.")
    else:
        st.warning("Models produced different predictions. Review interpretability section below.")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 4. INTERPRETATION (Grad-CAM Panel)
# =========================================================
if st.session_state.mel is not None:

    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("4. Model Interpretability")

    with st.spinner("Generating Grad-CAM heatmaps..."):
        x = torch.tensor(mel).unsqueeze(0).float()

        cam_tcn = GradCAM(pure_tcn)
        cam_snn = GradCAM(tcn_snn)

        cls_idx_tcn = LABELS.index(pred_tcn)
        cls_idx_snn = LABELS.index(pred_snn)

        heat_tcn = cam_tcn.generate(x, cls_idx_tcn)
        heat_snn = cam_snn.generate(x, cls_idx_snn)

    # Side-by-side GradCAM
    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(mel, aspect="auto", cmap="viridis", origin="lower")
        ax.set_title("Input Mel-Spectrogram")
        st.pyplot(fig, clear_figure=True)

    with c2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(heat_tcn, aspect="auto", cmap="inferno", origin="lower")
        ax.set_title("Pure TCN Attention")
        st.pyplot(fig, clear_figure=True)

    with c3:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(heat_snn, aspect="auto", cmap="inferno", origin="lower")
        ax.set_title("Hybrid TCN-SNN Attention")
        st.pyplot(fig, clear_figure=True)

    st.markdown("""
    <p style='color:#444; font-size:15px;'>
    Higher-intensity regions (toward red) indicate areas of the spectrogram that strongly influenced model decisions.
    These correspond to characteristic respiratory sound events described in the manuscript (e.g., crackles, wheezes).
    </p>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# RESET
# =========================================================
st.divider()
if st.button("Run another analysis"):
    st.session_state.clear()
    st.rerun()

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style="text-align:center; margin-top:32px; color:#9ca3af; font-size:16px;">
PulmoScope © 2025<br>
Academic prototype — not a medical device.
</div>
""", unsafe_allow_html=True)
