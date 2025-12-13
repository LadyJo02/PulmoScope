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
# CUSTOM CSS (Blue Theme + Animated Background + Modal)
# =========================================================
st.markdown("""
<style>

body {
    background: linear-gradient(-45deg, #e8f1f9, #f6f9fc, #dde9f5, #f4f7fa);
    background-size: 400% 400%;
    animation: clinicalFlow 12s ease infinite;
}

@keyframes clinicalFlow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.block-container { padding-top: 1.2rem; }

/* Medical blue buttons */
.stButton > button {
    background-color: #2563EB !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    border: none !important;
}
.stButton > button:hover {
    background-color: #1E3A8A !important;
}

/* Custom modal overlay */
#modal-bg {
    position: fixed;
    top:0; left:0;
    width:100%; height:100%;
    background: rgba(0,0,0,0.45);
    display:flex;
    justify-content:center;
    align-items:center;
    z-index:9999;
}

#modal-box {
    background:white;
    padding:28px;
    border-radius:12px;
    width:350px;
    box-shadow: 0px 4px 16px rgba(0,0,0,0.25);
    text-align:center;
    border:1px solid #E5E7EB;
}

.modal-btn {
    background-color: #2563EB;
    color:white;
    padding:8px 16px;
    margin:8px;
    border-radius:6px;
    border:none;
    cursor:pointer;
}
.modal-btn:hover {
    background-color:#1E40AF;
}

.modal-cancel {
    background-color:#6B7280;
}
.modal-cancel:hover {
    background-color:#4B5563;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# GLOBALS
# =========================================================
LABELS = ["COPD", "Healthy", "Pneumonia", "Other"]

if "mel" not in st.session_state:
    st.session_state.mel = None
if "show_modal" not in st.session_state:
    st.session_state.show_modal = False


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div style="text-align:center; margin-bottom:18px;">
    <h1 style="margin-bottom:0px; font-size:36px; color:#1F2937;">PulmoScope</h1>
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

c1, c2 = st.columns(2)
c1.info("Pure TCN model loaded successfully.")
c2.info("Hybrid TCN-SNN model loaded successfully.")

st.markdown("---")


# =========================================================
# INPUT SECTION
# =========================================================
st.subheader("1. Input Lung Sound")

input_mode = st.radio(
    "Choose input method:",
    ["Upload .wav file", "Record via microphone"],
    horizontal=True,
)

audio = None
left, right = st.columns([1,1])

with left:
    if input_mode == "Upload .wav file":
        up = st.file_uploader("Upload (.wav)", type=["wav"])
        if up:
            st.audio(up)
            audio = load_audio(up)

    else:
        rec = st.audio_input("Record lung sound")
        if rec:
            st.audio(rec)
            audio = load_audio(rec)

with right:
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


st.markdown("---")


# =========================================================
# ANALYSIS BUTTON WITH REAL WORKING MODAL
# =========================================================

# User clicks Analyze → show modal
if audio is not None:
    if st.button("Run PulmoScope Analysis"):
        st.session_state.show_modal = True


# ------------------------------ MODAL OVERLAY ------------------------------
if st.session_state.show_modal:

    # Show grey background overlay
    st.markdown("""
    <div style="
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: rgba(0,0,0,0.4);
        z-index: 9998;
    "></div>
    """, unsafe_allow_html=True)

    # Create modal box container
    modal_container = st.container()
    with modal_container:
        st.markdown("""
        <div style="
            position: fixed;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            width: 360px;
            padding: 24px;
            border-radius: 12px;
            text-align: center;
            z-index: 9999;
            border: 1px solid #E5E7EB;
            box-shadow: 0 4px 15px rgba(0,0,0,0.25);
        ">
        <h4>Proceed with Analysis?</h4>
        <p style="font-size: 14px; color: #4B5563;">
            PulmoScope will preprocess your audio and run both AI models.
        </p>
        </div>
        """, unsafe_allow_html=True)

        # Buttons MUST be below HTML for Streamlit to render them interactively
        colA, colB = st.columns(2)
        with colA:
            if st.button("Analyze Now"):
                st.session_state.mel = create_mel(audio)
                st.session_state.show_modal = False
                st.rerun()

        with colB:
            if st.button("Cancel"):
                st.session_state.show_modal = False
                st.rerun()



# =========================================================
# SPECTROGRAM + MODEL OUTPUTS
# =========================================================
mel = st.session_state.mel

if mel is not None:

    left, right = st.columns([1.1, 0.9])

    # ---------------- SPECTROGRAM ----------------
    with left:
        st.markdown("""
        <div style="background:white; padding:18px; border-radius:10px; border:1px solid #E5E7EB;">
            <h4>Mel-Spectrogram</h4>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6.5,4))
        ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        st.pyplot(fig, clear_figure=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- MODEL PREDICTIONS ----------------
    with right:
        st.markdown("""
        <div style="background:white; padding:18px; border-radius:10px; border:1px solid #E5E7EB;">
            <h4>Model Predictions and Confidence Scores</h4>
        """, unsafe_allow_html=True)

        pred_tcn, prob_tcn = predict(pure_tcn, mel)
        pred_snn, prob_snn = predict(tcn_snn, mel)

        c1, c2 = st.columns(2)

        with c1:
            st.write(f"**Pure TCN:** {pred_tcn}")
            for label, p in zip(LABELS, prob_tcn):
                st.write(f"{label}: {p*100:.1f}%")
                st.progress(float(p))

        with c2:
            st.write(f"**Hybrid TCN-SNN:** {pred_snn}")
            for label, p in zip(LABELS, prob_snn):
                st.write(f"{label}: {p*100:.1f}%")
                st.progress(float(p))

        if pred_tcn == pred_snn:
            st.info(f"Both models agree: **{pred_tcn}**")
        else:
            st.warning("Models disagree — review Grad-CAM.")

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.caption("Upload or record audio, then run analysis.")

st.markdown("---")


# =========================================================
# GRAD-CAM INTERPRETATION
# =========================================================
st.subheader("3. Model Interpretation (Grad-CAM)")

with st.expander("Show heatmaps"):
    if mel is not None:

        x = torch.tensor(mel).unsqueeze(0).float()
        cam_tcn = GradCAM(pure_tcn)
        cam_snn = GradCAM(tcn_snn)

        cls_tcn = LABELS.index(pred_tcn)
        cls_snn = LABELS.index(pred_snn)

        heat1 = cam_tcn.generate(x, cls_tcn)
        heat2 = cam_snn.generate(x, cls_snn)

        c1, c2, c3 = st.columns(3)

        with c1:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title("Input Mel")
            st.pyplot(fig, clear_figure=True)

        with c2:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.imshow(heat1, aspect="auto", origin="lower", cmap="inferno")
            ax.set_title("TCN Attention Map")
            st.pyplot(fig, clear_figure=True)

        with c3:
            fig, ax = plt.subplots(figsize=(4,3))
            aximshow = ax.imshow(heat2, aspect="auto", origin="lower", cmap="inferno")
            ax.set_title("TCN-SNN Attention Map")
            st.pyplot(fig, clear_figure=True)

    else:
        st.info("Run analysis to view interpretation.")

st.markdown("---")


# =========================================================
# RESET
# =========================================================
if st.button("Start a new analysis"):
    st.session_state.clear()
    st.rerun()


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style="text-align:center; margin-top:32px; color:#6B7280;">
PulmoScope © 2025<br>Academic prototype — not a medical device
</div>
""", unsafe_allow_html=True)
