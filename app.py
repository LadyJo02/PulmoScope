import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.preprocess import load_audio, create_mel
from utils.inference import load_model, predict
from utils.gradcam import GradCAM

# =========================================================
# GLOBAL PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="PulmoScope",
    layout="wide",
)

# Background (subtle clinical tone)
st.markdown("""
<style>
body { background-color: #F5F7FA; }
.block-container { padding-top: 1.5rem; }
h1, h2, h3, h4 { color: #1F2937; }
</style>
""", unsafe_allow_html=True)

LABELS = ["COPD", "Healthy", "Pneumonia", "Other"]

# Session defaults
if "mel" not in st.session_state:
    st.session_state.mel = None

# =========================================================
# HEADER — CLEAN CLINICAL PRESENTATION
# =========================================================
st.markdown("""
<div style="text-align:center; margin-bottom:18px;">
    <h1 style="margin-bottom:0px; font-size:36px;">PulmoScope</h1>
    <p style="color:#4B5563; font-size:16px; margin-top:4px;">
        AI-assisted auscultation analysis for respiratory disease screening
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS — CACHED FOR SPEED
# =========================================================
@st.cache_resource
def load_models():
    pure = load_model("models/pure_tcn_weights.pth", "tcn")
    snn = load_model("models/tcn_snn_weights.pth", "snn")
    return pure, snn

pure_tcn, tcn_snn = load_models()

load_c1, load_c2 = st.columns(2)
load_c1.info("Pure TCN model loaded successfully.")
load_c2.info("Hybrid TCN-SNN model loaded successfully.")

st.markdown("---")

# =========================================================
# SECTION 1 — AUDIO INPUT (CLEAN TWO-COLUMN LAYOUT)
# =========================================================
st.subheader("1. Input Lung Sound")

input_mode = st.radio(
    "Choose input method:",
    ["Upload .wav file", "Record via microphone"],
    horizontal=True,
)

audio = None

left_col, right_col = st.columns([1, 1])

with left_col:
    if input_mode == "Upload .wav file":
        uploaded = st.file_uploader("Upload a lung sound (.wav)", type=["wav"])
        if uploaded:
            st.audio(uploaded)
            audio = load_audio(uploaded)

    else:  # microphone mode
        recorded = st.audio_input("Record lung sound")
        if recorded:
            st.audio(recorded)
            audio = load_audio(recorded)

with right_col:
    st.markdown("""
    <div style="padding:18px; background:white; border-radius:10px; border:1px solid #E5E7EB;">
        <h4 style="margin-bottom:6px;">Clinical Notes for Acquisition</h4>
        <p style="font-size:14px; color:#4B5563;">
            Ensure a quiet environment and hold the microphone close to the chest.
            Avoid excessive movement or rubbing noise.
            A minimum of 2–3 seconds of steady breathing improves model reliability.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# SECTION 2 — PREPROCESSING & SPECTROGRAM
# =========================================================
st.subheader("2. Preprocessing and Feature Extraction")

if audio is not None:
    if st.button("Run PulmoScope Analysis", type="primary"):
        with st.spinner("Processing audio and generating spectrogram..."):
            st.session_state.mel = create_mel(audio)

mel = st.session_state.mel

if mel is not None:
    sp_left, sp_right = st.columns([1.1, 0.9])

    # Mel-spectrogram display
    with sp_left:
        st.markdown("""
        <div style="padding:18px; background:white; border-radius:10px; border:1px solid #E5E7EB;">
        <h4>Mel-Spectrogram Representation</h4>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_title("Extracted Mel-Spectrogram", fontsize=11)
        st.pyplot(fig, clear_figure=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # SECTION 3 — MODEL OUTPUTS (TCN vs TCN-SNN)
    # =====================================================
    with sp_right:
        st.markdown("""
        <div style="padding:18px; background:white; border-radius:10px; border:1px solid #E5E7EB;">
        <h4>Model Predictions and Confidence Scores</h4>
        """, unsafe_allow_html=True)

        pred_tcn, prob_tcn = predict(pure_tcn, mel)
        pred_snn, prob_snn = predict(tcn_snn, mel)

        col_tcn, col_snn = st.columns(2)

        # --- Pure TCN ---
        with col_tcn:
            st.write("Pure TCN Prediction:", f"**{pred_tcn}**")
            for label, score in zip(LABELS, prob_tcn):
                st.write(f"{label}: {score * 100:.1f}%")
                st.progress(float(score))

        # --- Hybrid TCN-SNN ---
        with col_snn:
            st.write("Hybrid TCN-SNN Prediction:", f"**{pred_snn}**")
            for label, score in zip(LABELS, prob_snn):
                st.write(f"{label}: {score * 100:.1f}%")
                st.progress(float(score))

        # Consensus Display
        if pred_tcn == pred_snn:
            st.info(f"Both models agree: predicted **{pred_tcn}**.")
        else:
            st.warning("Models disagree. Review the interpretability heatmaps below.")

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.caption("Please upload or record audio, then run the analysis.")

st.markdown("---")

# =========================================================
# SECTION 4 — INTERPRETABILITY (GRAD-CAM)
# =========================================================
st.subheader("3. Model Interpretation (Grad-CAM Attention Maps)")

with st.expander("Show interpretability heatmaps"):
    if mel is not None:

        x = torch.tensor(mel).unsqueeze(0).float()

        cam_tcn = GradCAM(pure_tcn)
        cam_snn = GradCAM(tcn_snn)

        cls_tcn = LABELS.index(pred_tcn)
        cls_snn = LABELS.index(pred_snn)

        heat_tcn = cam_tcn.generate(x, cls_tcn)
        heat_snn = cam_snn.generate(x, cls_snn)

        i1, i2, i3 = st.columns(3)

        # Input
        with i1:
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title("Input Mel-Spectrogram", fontsize=10)
            st.pyplot(fig, clear_figure=True)

        # TCN attention
        with i2:
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            ax.imshow(heat_tcn, aspect="auto", origin="lower", cmap="inferno")
            ax.set_title("Pure TCN Attention Map", fontsize=10)
            st.pyplot(fig, clear_figure=True)

        # SNN attention
        with i3:
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            ax.imshow(heat_snn, aspect="auto", origin="lower", cmap="inferno")
            ax.set_title("Hybrid TCN-SNN Attention Map", fontsize=10)
            st.pyplot(fig, clear_figure=True)

        st.markdown("""
        <p style="color:#4B5563; font-size:14px; margin-top:12px;">
        Highlighted regions indicate acoustic segments most influential to the model’s decision.
        Redder areas represent stronger activation, often corresponding to crackles, wheezes,
        or transient events known in respiratory analysis literature.
        </p>
        """, unsafe_allow_html=True)
    else:
        st.info("Run analysis to view interpretability results.")

st.markdown("---")

# =========================================================
# RESET BUTTON
# =========================================================
if st.button("Start a new analysis"):
    st.session_state.clear()
    st.rerun()

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style="text-align:center; margin-top:32px; color:#6B7280; font-size:15px;">
PulmoScope © 2025<br>
Academic prototype — not a medical device
</div>
""", unsafe_allow_html=True)
