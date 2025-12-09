import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import queue

from utils.preprocess import load_audio, create_mel
from utils.inference import load_model, predict

# =========================================================
# OPTIONAL REALTIME (SAFE IMPORT)
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
# LOAD MODELS (REAL MODELS ONLY)
# =========================================================
@st.cache_resource
def load_models():
    pure = load_model("models/pure_tcn_weights.pth", model_type="tcn")
    snn  = load_model("models/tcn_snn_weights.pth", model_type="snn")
    return pure, snn


pure_tcn, tcn_snn = load_models()

c1, c2 = st.columns(2)
if pure_tcn is not None:
    c1.success("✅ Pure TCN Loaded")
else:
    c1.error("❌ Failed to load Pure TCN model")

if tcn_snn is not None:
    c2.success("✅ Hybrid TCN-SNN Loaded")
else:
    c2.error("❌ Failed to load Hybrid TCN-SNN model")

st.divider()

# =========================================================
# OPTIONAL REALTIME AUDIO (DISABLED ON CLOUD)
# =========================================================
if USE_REALTIME:
    class AudioCollector(AudioProcessorBase):
        def __init__(self):
            self.buffer = queue.Queue()

        def recv_audio(self, frame):
            audio = frame.to_ndarray().flatten().astype(np.float32)
            self.buffer.put(audio)
            return frame

    with st.expander("🎙 Realtime Audio (Experimental)"):
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

            if len(frames) > 0:
                audio = np.concatenate(frames)
                mel = create_mel(audio)
                st.success("Realtime audio captured.")
            else:
                st.warning("No audio detected.")
else:
    st.info("🎙 Realtime recording is disabled on Streamlit Cloud.")

# =========================================================
# MAIN UI – SINGLE SCREEN
# =========================================================
left, right = st.columns([1, 1])
mel = None

# -------------------------
# LEFT: INPUT + SPECTROGRAM
# -------------------------
with left:
    st.subheader("Input")

    uploaded_file = st.file_uploader(
        "Upload lung sound recording (WAV)",
        type=["wav"]
    )

    if uploaded_file:
        st.audio(uploaded_file)
        audio = load_audio(uploaded_file)
        mel = create_mel(audio)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(mel, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title("Mel-Spectrogram")
        st.pyplot(fig, clear_figure=True)

# -------------------------
# RIGHT: MODEL COMPARISON
# -------------------------
with right:
    if mel is not None:
        st.subheader("Model Comparison")

        if pure_tcn is None and tcn_snn is None:
            st.error("Models failed to load – please check the weights in /models.")
        else:
            pred_tcn, prob_tcn = predict(pure_tcn, mel)
            pred_snn, prob_snn = predict(tcn_snn, mel)

            labels = ["Normal", "COPD", "Pneumonia", "Other"]

            a, b = st.columns(2)

            with a:
                st.markdown("### Pure TCN")
                st.success(pred_tcn)
                for l, p in zip(labels, prob_tcn):
                    st.write(f"{l}: {p*100:.1f}%")

            with b:
                st.markdown("### Hybrid TCN-SNN")
                st.success(pred_snn)
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
