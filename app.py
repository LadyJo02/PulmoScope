import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.preprocess import load_audio, create_mel
from utils.inference import load_model, predict
from utils.gradcam import GradCAM


# =====================================================================
# PAGE CONFIG + GLOBAL STYLING
# =====================================================================
st.set_page_config(
    page_title="PulmoScope",
    layout="wide",
)

st.markdown(
    """
    <style>
    body {
        background-color: #F5F7FA;
    }
    .card {
        padding: 18px;
        background: white;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 10px;
        color: #111827;
    }
    .subheader {
        font-size: 18px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 6px;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #6B7280;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

LABELS = ["COPD", "Healthy", "Pneumonia", "Other"]

# Session defaults
if "mel" not in st.session_state:
    st.session_state.mel = None

if "audio" not in st.session_state:
    st.session_state.audio = None


# =====================================================================
# HEADER
# =====================================================================
st.markdown(
    """
    <div style="text-align:center; margin-bottom:25px;">
        <h1 style="margin-bottom:4px;">PulmoScope</h1>
        <p style="color:#6b7280; font-size:16px;">
            AI-assisted analysis of lung auscultation recordings
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# LOAD MODELS
# =====================================================================
@st.cache_resource
def load_models():
    pure = load_model("models/pure_tcn_weights.pth", "tcn")
    snn = load_model("models/tcn_snn_weights.pth", "snn")
    return pure, snn


pure_tcn, tcn_snn = load_models()


# =====================================================================
# LAYOUT: LEFT (INPUT + SIGNAL VIEW) / RIGHT (PREDICTIONS)
# =====================================================================
left, right = st.columns([1.1, 1.1])


# =====================================================================
# LEFT COLUMN — AUDIO INPUT
# =====================================================================
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Step 1: Provide Lung Sound Input</div>", unsafe_allow_html=True)

    mode = st.radio(
        "Choose input method",
        ["Upload WAV file", "Record from microphone"],
        horizontal=True
    )

    audio = None

    # Upload
    if mode == "Upload WAV file":
        file = st.file_uploader("Upload lung sound (.wav)", type=["wav"])
        if file is not None:
            st.audio(file)
            audio = load_audio(file)

    # Record
    if mode == "Record from microphone":
        recorded_audio = st.audio_input("Record lung sound")
        if recorded_audio is not None:
            st.audio(recorded_audio)
            audio = load_audio(recorded_audio)

    if audio is not None:
        st.session_state.audio = audio

    # Analyze button
    if st.session_state.audio is not None:
        if st.button("Run Analysis", type="primary"):
            st.session_state.mel = create_mel(st.session_state.audio)

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# LEFT COLUMN — SIGNAL VISUALIZATION
# =====================================================================
if st.session_state.mel is not None:
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Step 2: Signal Representation</div>", unsafe_allow_html=True)

        mel = st.session_state.mel

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title("Mel-spectrogram", fontsize=12)
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.tick_params(labelsize=8)
        st.pyplot(fig, clear_figure=True)

        st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# RIGHT COLUMN — MODEL INFERENCE
# =====================================================================
pred_tcn = None
pred_snn = None
prob_tcn = None
prob_snn = None

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Step 3: Model Predictions</div>", unsafe_allow_html=True)

    if st.session_state.mel is not None:

        pred_tcn, prob_tcn = predict(pure_tcn, st.session_state.mel)
        pred_snn, prob_snn = predict(tcn_snn, st.session_state.mel)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='subheader'>Pure TCN</div>", unsafe_allow_html=True)
            st.write(f"Prediction: **{pred_tcn}**")
            for label, p in zip(LABELS, prob_tcn):
                st.write(f"{label}: {p*100:.1f}%")
                st.progress(float(p))

        with c2:
            st.markdown("<div class='subheader'>Hybrid TCN-SNN</div>", unsafe_allow_html=True)
            st.write(f"Prediction: **{pred_snn}**")
            for label, p in zip(LABELS, prob_snn):
                st.write(f"{label}: {p*100:.1f}%")
                st.progress(float(p))

    else:
        st.info("Upload or record audio to begin analysis.")

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# INTERPRETATION SECTION — GRAD-CAM
# =====================================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Model Interpretation</div>", unsafe_allow_html=True)

with st.expander("Show Grad-CAM Visualizations"):
    if st.session_state.mel is not None:

        x = torch.tensor(st.session_state.mel).unsqueeze(0).float()

        try:
            cam_tcn = GradCAM(pure_tcn)
            cam_snn = GradCAM(tcn_snn)

            idx_tcn = LABELS.index(pred_tcn)
            idx_snn = LABELS.index(pred_snn)

            heat_tcn = cam_tcn.generate(x, idx_tcn)
            heat_snn = cam_snn.generate(x, idx_snn)

            g1, g2 = st.columns(2)

            with g1:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.imshow(heat_tcn, aspect="auto", origin="lower", cmap="inferno")
                ax.set_title("Pure TCN Activation Map", fontsize=12)
                st.pyplot(fig, clear_figure=True)

            with g2:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.imshow(heat_snn, aspect="auto", origin="lower", cmap="inferno")
                ax.set_title("Hybrid TCN-SNN Activation Map", fontsize=12)
                st.pyplot(fig, clear_figure=True)

        except Exception:
            st.warning("Grad-CAM could not be generated for this sample.")

    else:
        st.info("Run analysis first to generate explanation maps.")

st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# RESET BUTTON
# =====================================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)

if st.button("Analyze Another Recording"):
    st.session_state.clear()
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# FOOTER
# =====================================================================
st.markdown(
    """
    <div class="footer">
        PulmoScope © 2025<br>
        Academic prototype — not a medical device
    </div>
    """,
    unsafe_allow_html=True
)
