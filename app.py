import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.preprocess import load_audio, create_mel
from utils.inference import load_model, predict
from utils.gradcam import GradCAM

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="PulmoScope",
    page_icon="🫁",
    layout="wide",
)

LABELS = ["COPD", "Healthy", "Pneumonia", "Other"]

# Session defaults
if "mel" not in st.session_state:
    st.session_state.mel = None

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div style="text-align:center; margin-bottom:28px;">
        <h1 style="margin-bottom:4px;">PulmoScope</h1>
        <p style="color:#6b7280; font-size:15px;">
            AI-assisted respiratory disease analysis from lung sound recordings
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    pure = load_model("models/pure_tcn_weights.pth", "tcn")
    snn = load_model("models/tcn_snn_weights.pth", "snn")
    return pure, snn


pure_tcn, tcn_snn = load_models()

m1, m2 = st.columns(2)
m1.info(f"Pure TCN: {'Loaded' if pure_tcn else 'Failed'}")
m2.info(f"Hybrid TCN-SNN: {'Loaded' if tcn_snn else 'Failed'}")

st.divider()

# =========================================================
# AUDIO INPUT
# =========================================================
st.subheader("Audio Input")

mode = st.radio(
    "Select input method",
    ["Upload WAV file", "Record from microphone"],
    horizontal=True,
)

audio = None

# ---------- Upload ----------
if mode == "Upload WAV file":
    file = st.file_uploader("Upload lung sound (.wav)", type=["wav"])
    if file is not None:
        st.audio(file)
        audio = load_audio(file)

# ---------- Record ----------
if mode == "Record from microphone":
    recorded_audio = st.audio_input("Record lung sound")
    if recorded_audio is not None:
        st.audio(recorded_audio)
        audio = load_audio(recorded_audio)

# =========================================================
# ANALYSIS
# =========================================================
if audio is not None and st.button("Analyze recording", type="primary"):
    st.session_state.mel = create_mel(audio)

mel = st.session_state.mel

# Prepare vars for explanation section
pred_tcn = None
pred_snn = None
prob_tcn = None
prob_snn = None

# =========================================================
# RESULTS
# =========================================================
if mel is not None:
    left, right = st.columns([1.1, 0.9])

    with left:
        st.subheader("Signal Representation")
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title("Mel-spectrogram", fontsize=11)
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.tick_params(labelsize=8)
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Model Comparison")

        pred_tcn, prob_tcn = predict(pure_tcn, mel)
        pred_snn, prob_snn = predict(tcn_snn, mel)

        c1, c2 = st.columns(2)

        for title, pred, prob, col in [
            ("Pure TCN", pred_tcn, prob_tcn, c1),
            ("Hybrid TCN-SNN", pred_snn, prob_snn, c2),
        ]:
            with col:
                st.markdown(f"**{title}**")
                st.markdown(f"Prediction: **{pred}**")
                for label, p in zip(LABELS, prob):
                    st.write(f"{label}: {p * 100:.1f}%")
                    st.progress(float(p))

else:
    st.caption("Upload or record audio, then click Analyze recording.")

# =========================================================
# EXPLANATION (SIDE-BY-SIDE)
# =========================================================
st.divider()

with st.expander("Model explanation (input + attention comparison)"):
    if mel is not None and pure_tcn is not None and tcn_snn is not None and pred_tcn is not None:
        try:
            x = torch.tensor(mel).unsqueeze(0).float()

            cam_tcn = GradCAM(pure_tcn)
            cam_snn = GradCAM(tcn_snn)

            cls_idx_tcn = LABELS.index(pred_tcn) if pred_tcn in LABELS else 0
            cls_idx_snn = LABELS.index(pred_snn) if pred_snn in LABELS else 0

            heat_tcn = cam_tcn.generate(x, cls_idx_tcn)
            heat_snn = cam_snn.generate(x, cls_idx_snn)

            c1, c2, c3 = st.columns(3)

            with c1:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.imshow(mel, aspect="auto", cmap="viridis", origin="lower")
                ax.set_title("Input Mel", fontsize=10)
                ax.tick_params(labelsize=8)
                st.pyplot(fig, clear_figure=True)

            with c2:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.imshow(heat_tcn, aspect="auto", cmap="inferno", origin="lower")
                ax.set_title("Pure TCN Attention", fontsize=10)
                ax.tick_params(labelsize=8)
                st.pyplot(fig, clear_figure=True)

            with c3:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.imshow(heat_snn, aspect="auto", cmap="inferno", origin="lower")
                ax.set_title("Hybrid TCN-SNN Attention", fontsize=10)
                ax.tick_params(labelsize=8)
                st.pyplot(fig, clear_figure=True)

        except Exception:
            st.warning("Model explanation unavailable for this sample.")
    else:
        st.info("Run analysis first to view model explanations.")

# =========================================================
# RESET
# =========================================================
st.divider()
if st.button("Analyze another recording"):
    st.session_state.clear()
    st.rerun()

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div style="text-align:center; margin-top:32px; color:#9ca3af; font-size:16px;">
        PulmoScope © 2025<br>
        Academic prototype — not a medical device
    </div>
    """,
    unsafe_allow_html=True,
)
