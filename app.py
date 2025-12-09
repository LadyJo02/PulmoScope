import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import queue

from utils.preprocess import load_audio, create_mel
from utils.inference import load_model, predict
from utils.gradcam import GradCAM


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="PulmoScope",
    page_icon="🫁",
    layout="wide"
)

LABELS = ["COPD", "Healthy", "Pneumonia", "Other"]


# =========================================================
# OPTIONAL REALTIME AUDIO
# =========================================================
USE_REALTIME = True
try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
except Exception:
    USE_REALTIME = False


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div style="text-align:center; margin-bottom:28px;">
        <h1>PulmoScope</h1>
        <p style="color:#6b7280;">
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
    return (
        load_model("models/pure_tcn_weights.pth", "tcn"),
        load_model("models/tcn_snn_weights.pth", "snn"),
    )


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
    ["Upload WAV file", "Realtime recording (experimental)"],
    horizontal=True,
)

audio = None

if mode == "Upload WAV file":
    file = st.file_uploader("Upload lung sound (.wav)", type=["wav"])
    if file:
        st.audio(file)
        audio = load_audio(file)

if mode == "Realtime recording (experimental)":
    st.caption("Realtime recording may not work on all browsers.")

    if USE_REALTIME:
        class AudioCollector(AudioProcessorBase):
            def __init__(self):
                self.buffer = queue.Queue()

            def recv_audio(self, frame):
                self.buffer.put(frame.to_ndarray().flatten().astype(np.float32))
                return frame

        ctx = webrtc_streamer(
            key="audio",
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


# =========================================================
# ANALYSIS
# =========================================================
mel = None

if audio is not None and st.button("Analyze recording", type="primary"):
    mel = create_mel(audio)


# =========================================================
# RESULTS
# =========================================================
if mel is not None:
    left, right = st.columns([1.1, 0.9])

    with left:
        st.subheader("Signal Representation")
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
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
                for l, p in zip(LABELS, prob):
                    st.write(f"{l}: {p*100:.1f}%")
                    st.progress(float(p))


# =========================================================
# EXPLANATION (SIDE-BY-SIDE)
# =========================================================
st.divider()

with st.expander("Model explanation (input + attention comparison)"):
    if mel is not None and pure_tcn and tcn_snn:
        try:
            x = torch.tensor(mel).unsqueeze(0).float()

            cam_tcn = GradCAM(pure_tcn)
            cam_snn = GradCAM(tcn_snn)

            heat_tcn = cam_tcn.generate(x, LABELS.index(pred_tcn))
            heat_snn = cam_snn.generate(x, LABELS.index(pred_snn))

            c1, c2, c3 = st.columns(3)

            with c1:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.imshow(mel, aspect="auto", cmap="viridis", origin="lower")
                ax.set_title("Input Mel")
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

        except Exception as e:
            st.warning("Model explanation unavailable for this sample.")


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
