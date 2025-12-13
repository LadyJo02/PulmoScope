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

# =========================================================
# CLINICAL UI STYLING (BLUE BUTTONS + ANIMATED BACKGROUND)
# =========================================================
st.markdown("""
<style>

body {
    background: linear-gradient(120deg, #e6f3ff 0%, #f5faff 100%);
    animation: gradientShift 6s ease infinite alternate;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}

button[kind="primary"] {
    background-color: #2563EB !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    border: none !important;
}
button {
    border-radius: 8px !important;
}

.block-container { padding-top: 1.5rem; }
h1, h2, h3, h4 { color: #1F2937; }

</style>
""", unsafe_allow_html=True)

LABELS = ["COPD", "Healthy", "Pneumonia", "Other"]

# =========================================================
# SESSION DEFAULTS
# =========================================================
if "mel" not in st.session_state:
    st.session_state.mel = None

if "audio_ready" not in st.session_state:
    st.session_state.audio_ready = False

# =========================================================
# HEADER
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
# LOAD MODELS
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
# SECTION 1 — INPUT
# =========================================================
st.subheader("1. Input Lung Sound")

input_mode = st.radio(
    "Choose input method:",
    ["Upload .wav file", "Record via microphone"],
    horizontal=True,
)

audio = None

# --- Layout ---
left_col, right_col = st.columns([1, 1])

with left_col:
    if input_mode == "Upload .wav file":
        uploaded = st.file_uploader("Upload lung sound (.wav)", type=["wav"])
        if uploaded:
            st.audio(uploaded)
            audio = load_audio(uploaded)
            st.session_state.audio_ready = True

    else:
        recorded = st.audio_input("Record lung sound")
        if recorded:
            st.audio(recorded)
            audio = load_audio(recorded)
            st.session_state.audio_ready = True

with right_col:
    st.markdown("""
    <div style="padding:18px; background:white; border-radius:10px; border:1px solid #E5E7EB;">
        <h4 style="margin-bottom:6px;">Clinical Notes</h4>
        <p style="font-size:14px; color:#4B5563;">
            • Record in a quiet environment.<br>
            • Keep the microphone near the chest, avoiding contact noise.<br>
            • Capture at least 5–10 seconds of steady respiration.<br>
            • Avoid rubbing, fabric noise, or speaking during recording.<br>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# POPUP DIALOG FOR ANALYSIS CONFIRMATION
# =========================================================
@st.dialog("Confirm PulmoScope Analysis")
def analysis_popup(audio):
    st.write(
        "Proceed with analysis? PulmoScope will extract features, "
        "compute predictions using TCN and TCN-SNN models, and generate interpretability maps."
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Proceed", type="primary"):
            st.session_state.mel = create_mel(audio)
            st.rerun()
    with c2:
        if st.button("Cancel"):
            st.session_state.mel = None
            st.rerun()

# =========================================================
# ANALYZE BUTTON (TRIGGERS POPUP)
# =========================================================
if st.session_state.audio_ready:
    if st.button("Run PulmoScope Analysis", type="primary"):
        analysis_popup(audio)
else:
    st.caption("Upload or record audio to enable analysis.")

mel = st.session_state.mel

# =========================================================
# SECTION 2 — SPECTROGRAM + MODEL PREDICTIONS
# =========================================================
if mel is not None:

    sp_left, sp_right = st.columns([1.1, 0.9])

    # --- Spectrogram ---
    with sp_left:
        st.markdown("""
        <div style="padding:18px; background:white; border-radius:10px; border:1px solid #E5E7EB;">
            <h4>Mel-Spectrogram</h4>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        st.pyplot(fig, clear_figure=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Predictions ---
    with sp_right:
        st.markdown("""
        <div style="padding:18px; background:white; border-radius:10px; border:1px solid #E5E7EB;">
            <h4>Model Predictions</h4>
        """, unsafe_allow_html=True)

        pred_tcn, prob_tcn = predict(pure_tcn, mel)
        pred_snn, prob_snn = predict(tcn_snn, mel)

        col_tcn, col_snn = st.columns(2)

        with col_tcn:
            st.write("Pure TCN:", f"**{pred_tcn}**")
            for label, score in zip(LABELS, prob_tcn):
                st.write(f"{label}: {score*100:.1f}%")
                st.progress(float(score))

        with col_snn:
            st.write("Hybrid TCN-SNN:", f"**{pred_snn}**")
            for label, score in zip(LABELS, prob_snn):
                st.write(f"{label}: {score*100:.1f}%")
                st.progress(float(score))

        if pred_tcn == pred_snn:
            st.success(f"Both models agree on **{pred_tcn}**.")
        else:
            st.info("Models differ. Consult the interpretability heatmaps below.")

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.caption("Awaiting analysis...")

st.markdown("---")

# =========================================================
# SECTION 3 — GRAD-CAM INTERPRETABILITY
# =========================================================
st.subheader("3. Model Interpretation")

with st.expander("Show attention heatmaps"):
    if mel is not None:

        x = torch.tensor(mel).unsqueeze(0).float()

        cam_tcn = GradCAM(pure_tcn)
        cam_snn = GradCAM(tcn_snn)

        heat_tcn = cam_tcn.generate(x, LABELS.index(pred_tcn))
        heat_snn = cam_snn.generate(x, LABELS.index(pred_snn))

        c1, c2, c3 = st.columns(3)

        with c1:
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            ax.imshow(mel, cmap="viridis", origin="lower")
            ax.set_title("Input Mel")
            st.pyplot(fig, clear_figure=True)

        with c2:
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            ax.imshow(heat_tcn, cmap="inferno", origin="lower")
            ax.set_title("TCN Attention")
            st.pyplot(fig, clear_figure=True)

        with c3:
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            ax.imshow(heat_snn, cmap="inferno", origin="lower")
            ax.set_title("TCN-SNN Attention")
            st.pyplot(fig, clear_figure=True)

    else:
        st.info("No analysis performed.")

st.markdown("---")

# =========================================================
# NEW ANALYSIS POPUP
# =========================================================
@st.dialog("Start New Analysis")
def reset_popup():
    st.write("Are you sure you want to reset and begin a new analysis?")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Yes", type="primary"):
            st.session_state.clear()
            st.rerun()
    with colB:
        if st.button("Cancel"):
            st.rerun()

if st.button("Start a New Analysis"):
    reset_popup()

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style="text-align:center; margin-top:32px; color:#6B7280; font-size:15px;">
PulmoScope © 2025<br>
Academic prototype — not a medical device
</div>
""", unsafe_allow_html=True)
