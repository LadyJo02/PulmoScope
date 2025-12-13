import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.preprocess import load_audio, create_mel
from utils.inference import load_model, predict
from utils.gradcam import GradCAM

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PulmoScope",
    page_icon="🫁",
    layout="wide",
)

# =========================================================
# GLOBAL CLINICAL THEME + ANIMATED BACKGROUND
# =========================================================
st.markdown("""
<style>

/* Animated medical gradient background */
body {
    background: linear-gradient(120deg, #EAF2FB, #F5F9FF, #EAF2FB);
    background-size: 400% 400%;
    animation: gradientMove 18s ease infinite;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.block-container {
    padding-top: 1.5rem;
}

/* Typography */
h1, h2, h3, h4 {
    color: #0F172A;
    font-weight: 600;
}

/* Card style */
.card {
    background: white;
    border-radius: 12px;
    padding: 18px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
}

/* === DEFAULT BUTTON STYLE: MEDICAL BLUE === */
div.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1.2rem;
    font-weight: 500;
}

div.stButton > button:hover {
    background-color: #1D4ED8;
}

</style>
""", unsafe_allow_html=True)

LABELS = ["COPD", "Healthy", "Pneumonia", "Other"]

# =========================================================
# SESSION STATE
# =========================================================
if "mel" not in st.session_state:
    st.session_state.mel = None

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div style="text-align:center; margin-bottom:22px;">
    <h1 style="font-size:36px; margin-bottom:4px;">PulmoScope</h1>
    <p style="font-size:16px; color:#334155;">
        AI-assisted lung sound analysis for respiratory disease screening
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

c1, c2 = st.columns(2)
c1.info("Pure TCN model loaded")
c2.info("Hybrid TCN-SNN model loaded")

st.divider()

# =========================================================
# SECTION 1 — INPUT
# =========================================================
st.subheader("1. Lung Sound Acquisition")

mode = st.radio(
    "Select input method:",
    ["Upload .wav file", "Record via microphone"],
    horizontal=True,
)

audio = None
lcol, rcol = st.columns([1, 1])

with lcol:
    if mode == "Upload .wav file":
        uploaded = st.file_uploader("Upload lung sound (.wav)", type=["wav"])
        if uploaded:
            st.audio(uploaded)
            audio = load_audio(uploaded)
    else:
        recorded = st.audio_input("Record lung sound")
        if recorded:
            st.audio(recorded)
            audio = load_audio(recorded)

# EXACT clinical notes block
with rcol:
    st.markdown("""
        <div style="background:white; padding:18px; border-radius:10px; border:1px solid #E5E7EB;">
            <h4>Clinical Notes for Auscultation</h4>
            <p style="font-size:14px; color:#4B5563;">
                Record in a quiet environment, <br>
                minimize rubbing noise, <br>
                and capture at least 5–10 seconds of steady breathing.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.divider()

# =========================================================
# SECTION 2 — PREPROCESSING
# =========================================================
st.subheader("2. Preprocessing and Feature Extraction")

if audio is not None:
    if st.button("Run PulmoScope Analysis"):
        with st.spinner("Processing audio and generating spectrogram..."):
            st.session_state.mel = create_mel(audio)

mel = st.session_state.mel

# =========================================================
# SECTION 3 — RESULTS
# =========================================================
if mel is not None:
    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown("<div class='card'><h4>Mel-Spectrogram</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        st.pyplot(fig, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><h4>Model Predictions and Confidence</h4>", unsafe_allow_html=True)

        pred_tcn, prob_tcn = predict(pure_tcn, mel)
        pred_snn, prob_snn = predict(tcn_snn, mel)

        c1, c2 = st.columns(2)

        with c1:
            st.write("Pure TCN:", f"**{pred_tcn}**")
            for lbl, p in zip(LABELS, prob_tcn):
                st.write(f"{lbl}: {p*100:.1f}%")
                st.progress(float(p))

        with c2:
            st.write("Hybrid TCN-SNN:", f"**{pred_snn}**")
            for lbl, p in zip(LABELS, prob_snn):
                st.write(f"{lbl}: {p*100:.1f}%")
                st.progress(float(p))

        if pred_tcn == pred_snn:
            st.info(f"Model agreement: both predict **{pred_tcn}**.")
        else:
            st.warning("Model disagreement observed. Interpretability review recommended.")

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.caption("Upload or record audio to begin analysis.")

st.divider()

# =========================================================
# SECTION 4 — INTERPRETABILITY
# =========================================================
st.subheader("3. Model Interpretation (Grad-CAM)")

with st.expander("View attention heatmaps"):
    if mel is not None:
        x = torch.tensor(mel).unsqueeze(0).float()

        cam_tcn = GradCAM(pure_tcn)
        cam_snn = GradCAM(tcn_snn)

        cls_tcn = LABELS.index(pred_tcn)
        cls_snn = LABELS.index(pred_snn)

        heat_tcn = cam_tcn.generate(x, cls_tcn)
        heat_snn = cam_snn.generate(x, cls_snn)

        c1, c2, c3 = st.columns(3)

        with c1:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title("Input Mel-Spectrogram")
            st.pyplot(fig, clear_figure=True)

        with c2:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(heat_tcn, aspect="auto", origin="lower", cmap="inferno")
            ax.set_title("Pure TCN Attention")
            st.pyplot(fig, clear_figure=True)

        with c3:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(heat_snn, aspect="auto", origin="lower", cmap="inferno")
            ax.set_title("Hybrid TCN-SNN Attention")
            st.pyplot(fig, clear_figure=True)

        st.markdown("""
        <p style="font-size:14px; color:#475569;">
        <strong>How to interpret the heatmaps:</strong><br>
        Warmer colors (yellow to red) indicate time–frequency regions of the lung sound
        that contributed most strongly to the model’s prediction. Cooler colors indicate
        regions with minimal influence. Concentrated high-activation regions often
        correspond to clinically relevant acoustic events such as crackles, wheezes,
        or abnormal airflow patterns observed during auscultation.
        </p>
        """, unsafe_allow_html=True)

    else:
        st.info("Run analysis to enable interpretability.")

st.divider()

# =========================================================
# RESET
# =========================================================
if st.button("Start New Analysis"):
    st.session_state.clear()
    st.rerun()

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style="text-align:center; margin-top:32px; font-size:14px; color:#64748B;">
PulmoScope © 2025<br>
Academic prototype — not a medical device
</div>
""", unsafe_allow_html=True)
