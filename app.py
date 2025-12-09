import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils.preprocess import load_audio, create_mel
from utils.inference import load_model, predict

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import sounddevice as sd
import queue

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PulmoScope",
    page_icon="🫁",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================
st.image("assets/banner.png", use_container_width=True)

st.markdown(
    """
    <h1 style='text-align:center; color:#123C4C;'>PULMOSCOPE</h1>
    <p style='text-align:center; font-size:18px;'>
        AI-assisted respiratory disease analysis
    </p>
    """,
    unsafe_allow_html=True
)

# =========================================================
# LOAD MODELS (REAL – NO MOCK)
# =========================================================
@st.cache_resource
def load_models():
    pure = load_model("models/pure_tcn_weights.pth", model_type="tcn")
    snn  = load_model("models/tcn_snn_weights.pth", model_type="snn")
    return pure, snn

pure_tcn, tcn_snn = load_models()

status_col1, status_col2 = st.columns(2)
status_col1.success("✅ Pure TCN loaded")
status_col2.success("✅ Hybrid TCN-SNN loaded")

st.divider()

# =========================================================
# REALTIME AUDIO (OPTIONAL – SAFE IMPLEMENTATION)
# =========================================================
class AudioCollector(AudioProcessorBase):
    def __init__(self):
        self.buffer = queue.Queue()

    def recv_audio(self, frame):
        audio = frame.to_ndarray().flatten().astype(np.float32)
        self.buffer.put(audio)
        return frame

with st.expander("🎙 Realtime Recording (Optional)"):
    ctx = webrtc_streamer(
        key="pulmoscope-audio",
        audio_processor_factory=AudioCollector,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True
    )

    if ctx.audio_processor and st.button("Analyze Realtime Audio"):
        frames = []
        while not ctx.audio_processor.buffer.empty():
            frames.append(ctx.audio_processor.buffer.get())

        if len(frames) == 0:
            st.warning("No audio captured.")
        else:
            audio = np.concatenate(frames)
            mel = create_mel(audio)
            st.success("Realtime audio captured ✅")

# =========================================================
# MAIN UI – SINGLE SCREEN
# =========================================================
left, right = st.columns([1, 1])

# -------------------------
# LEFT: INPUT + SPECTROGRAM
# -------------------------
with left:
    st.subheader("Input")

    uploaded_file = st.file_uploader(
        "Upload lung sound recording (WAV)",
        type=["wav"]
    )

    mel = None

    if uploaded_file:
        st.audio(uploaded_file)
        audio = load_audio(uploaded_file)
        mel = create_mel(audio)

    if mel is not None:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(mel, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title("Mel-Spectrogram")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        st.pyplot(fig, clear_figure=True)

# -------------------------
# RIGHT: MODEL COMPARISON
# -------------------------
with right:
    if mel is not None:
        st.subheader("Model Comparison")

        pred_tcn, prob_tcn = predict(pure_tcn, mel)
        pred_snn, prob_snn = predict(tcn_snn, mel)

        colA, colB = st.columns(2)

        labels = ["Normal", "COPD", "Pneumonia", "Other"]

        with colA:
            st.markdown("### Pure TCN")
            st.success(f"Prediction: **{pred_tcn}**")
            for l, p in zip(labels, prob_tcn):
                st.write(f"{l}: {p*100:.1f}%")

        with colB:
            st.markdown("### Hybrid TCN-SNN")
            st.success(f"Prediction: **{pred_snn}**")
            for l, p in zip(labels, prob_snn):
                st.write(f"{l}: {p*100:.1f}%")

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.markdown(
    """
    <p style='text-align:center; color:gray; font-size:12px;'>
        PulmoScope © 2025<br>
        Academic Prototype – Not a Medical Device
    </p>
    """,
    unsafe_allow_html=True
)
