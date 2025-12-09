import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import queue

from utils.preprocess import load_audio, create_mel
from utils.inference import load_model, predict
from utils.gradcam import GradCAM


# =========================================================
# OPTIONAL REALTIME AUDIO
# =========================================================
USE_REALTIME = True
try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
except Exception:
    USE_REALTIME = False


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PulmoScope",
    page_icon="🫁",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div style="text-align:center; margin-bottom:32px;">
        <h1 style="margin-bottom:6px;">PulmoScope</h1>
        <p style="color:#6b7280; font-size:15px;">
            AI-assisted respiratory disease analysis from lung sound recordings
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CONSTANTS
# =========================================================
LABELS = ["COPD", "Healthy", "Pneumonia", "Other"]


# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    pure = load_model("models/pure_tcn_weights.pth", model_type="tcn")
    snn = load_model("models/tcn_snn_weights.pth", model_type="snn")
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

input_mode = st.radio(
    "Select input method",
    ["Upload WAV file", "Realtime recording (experimental)"],
    horizontal=True,
)

audio = None


# ---------- Upload ----------
if input_mode == "Upload WAV file":
    uploaded_file = st.file_uploader(
        "Upload lung sound recording (.wav)",
        type=["wav"],
    )

    if uploaded_file:
        st.audio(uploaded_file)
        audio = load_audio(uploaded_file)


# ---------- Realtime ----------
if input_mode == "Realtime recording (experimental)":
    st.info("Realtime recording may not be supported on Streamlit Cloud.")

    if USE_REALTIME:
        class AudioCollector(AudioProcessorBase):
            def __init__(self):
                self.buffer = queue.Queue()

            def recv_audio(self, frame):
                self.buffer.put(frame.to_ndarray().flatten().astype(np.float32))
                return frame

        ctx = webrtc_streamer(
            key="pulmoscope-audio",
            audio_processor_factory=AudioCollector,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
        )

        if ctx.audio_processor and st.button("Use recorded audio"):
            frames = []
            while not ctx.audio_processor.buffer.empty():
                frames.append(ctx.audio_processor.buffer.get())

            if frames:
                audio = np.concatenate(frames)
                st.success("Audio captured successfully.")
            else:
                st.warning("No audio detected.")


# =========================================================
# ANALYSIS
# =========================================================
mel = None

if audio is not None:
    st.divider()
    if st.button("Analyze recording", type="primary"):
        mel = create_mel(audio)
else:
    st.caption("Upload or record audio, then click Analyze recording.")


# =========================================================
# RESULTS
# =========================================================
if mel is not None:
    left, right = st.columns([1.1, 0.9])

    # ---------- MEL ----------
    with left:
        st.subheader("Representation")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.imshow(mel, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title("Mel-spectrogram", fontsize=11)
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.tick_params(labelsize=8)
        st.pyplot(fig, clear_figure=True)

    # ---------- PREDICTIONS ----------
    with right:
        st.subheader("Model Comparison")

        pred_tcn, prob_tcn = predict(pure_tcn, mel)
        pred_snn, prob_snn = predict(tcn_snn, mel)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Pure TCN**")
            st.markdown(f"Prediction: **{pred_tcn}**")
            for l, p in zip(LABELS, prob_tcn):
                st.write(f"{l}: {p*100:.1f}%")
                st.progress(float(p))

        with c2:
            st.markdown("**Hybrid TCN-SNN**")
            st.markdown(f"Prediction: **{pred_snn}**")
            for l, p in zip(LABELS, prob_snn):
                st.write(f"{l}: {p*100:.1f}%")
                st.progress(float(p))


# =========================================================
# GRAD-CAM EXPLANATION (SIDE-BY-SIDE)
# =========================================================
st.divider()

with st.expander("Model explanation (input + attention comparison)", expanded=False):

    if mel is None:
        st.info("Run analysis first to view explanations.")
    else:
        cls_tcn = LABELS.index(pred_tcn)
        cls_snn = LABELS.index(pred_snn)

        x_tensor = torch.tensor(mel).unsqueeze(0).float()

        cam_tcn = GradCAM(pure_tcn)
        cam_snn = GradCAM(tcn_snn)

        heat_tcn = cam_tcn.generate(x_tensor, cls_tcn)
        heat_snn = cam_snn.generate(x_tensor, cls_snn)

        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(mel, origin="lower", aspect="auto", cmap="viridis")
            ax.set_title("Input Mel", fontsize=10)
            ax.tick_params(labelsize=8)
            st.pyplot(fig, clear_figure=True)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(heat_tcn, origin="lower", aspect="auto", cmap="inferno")
            ax.set_title("Pure TCN — Grad-CAM", fontsize=10)
            ax.tick_params(labelsize=8)
            st.pyplot(fig, clear_figure=True)

        with col3:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(heat_snn, origin="lower", aspect="auto", cmap="inferno")
            ax.set_title("Hybrid TCN-SNN — Grad-CAM", fontsize=10)
            ax.tick_params(labelsize=8)
            st.pyplot(fig, clear_figure=True)


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
    <div style="text-align:center; margin-top:32px; color:#9ca3af; font-size:12px;">
        PulmoScope © 2025<br>
        Academic prototype — not a medical device
    </div>
    """,
    unsafe_allow_html=True,
)
