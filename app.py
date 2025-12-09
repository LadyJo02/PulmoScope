import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import queue

from utils.preprocess import load_audio, create_mel
from utils.inference import load_model, predict
from utils.gradcam import GradCAM


# ---------------------------------------------------------
# Optional realtime audio (may not work on all deployments)
# ---------------------------------------------------------
USE_REALTIME = True
try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
except Exception:
    USE_REALTIME = False


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="PulmoScope",
    layout="wide"
)

# ---------------------------------------------------------
# TITLE / HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; margin-bottom:40px;">
        <h1 style="margin-bottom:8px;">PulmoScope</h1>
        <p style="color:#6b7280; font-size:16px;">
            AI-assisted respiratory disease analysis from lung sound recordings
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    pure = load_model("models/pure_tcn_weights.pth", model_type="tcn")
    snn = load_model("models/tcn_snn_weights.pth", model_type="snn")
    return pure, snn


pure_tcn, tcn_snn = load_models()

status_left, status_right = st.columns(2)
status_left.info(
    "Model: Pure TCN\n\nStatus: {}".format("Loaded" if pure_tcn else "Failed")
)
status_right.info(
    "Model: Hybrid TCN-SNN\n\nStatus: {}".format("Loaded" if tcn_snn else "Failed")
)

st.divider()

# ---------------------------------------------------------
# AUDIO INPUT SECTION
# ---------------------------------------------------------
st.subheader("Audio Input")

input_mode = st.radio(
    "Select input method",
    ["Upload WAV file", "Realtime recording (experimental)"],
    horizontal=True,
)

audio = None  # raw waveform (numpy array)


# ---------- Upload mode ----------
if input_mode == "Upload WAV file":
    uploaded_file = st.file_uploader(
        "Upload lung sound recording (.wav)",
        type=["wav"],
        key="uploader",
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)
        audio = load_audio(uploaded_file)


# ---------- Realtime mode ----------
if input_mode == "Realtime recording (experimental)":
    st.info(
        "Realtime recording may not be supported on all browsers or cloud deployments. "
        "If recording does not work, please use WAV upload instead."
    )

    if USE_REALTIME:
        class AudioCollector(AudioProcessorBase):
            def __init__(self):
                self.buffer = queue.Queue()

            def recv_audio(self, frame):
                arr = frame.to_ndarray().flatten().astype(np.float32)
                self.buffer.put(arr)
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
                st.error("No audio detected. Please try recording again.")
    else:
        st.error("Realtime recording is not available in this environment.")


# ---------------------------------------------------------
# ANALYSIS TRIGGER
# ---------------------------------------------------------
mel = None

if audio is not None:
    st.divider()
    if st.button("Analyze recording", type="primary"):
        mel = create_mel(audio)
else:
    st.caption("Upload or record audio, then click Analyze recording.")


# ---------------------------------------------------------
# RESULTS: SPECTROGRAM + MODEL COMPARISON
# ---------------------------------------------------------
if mel is not None:
    left_col, right_col = st.columns([1.1, 0.9])

    # --- Left: Mel spectrogram ---
    with left_col:
        st.subheader("Representation")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(mel, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title("Mel-spectrogram")
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Frequency bins")
        st.pyplot(fig, clear_figure=True)

    # --- Right: Predictions ---
    with right_col:
        st.subheader("Model Comparison")

        if pure_tcn is None and tcn_snn is None:
            st.error("Models failed to load. Please check the weights in the models folder.")
        else:
            LABELS = ["COPD", "Healthy", "Pneumonia", "Other"]

            pred_tcn, prob_tcn = predict(pure_tcn, mel) if pure_tcn else ("Model not loaded", np.zeros(4))
            pred_snn, prob_snn = predict(tcn_snn, mel) if tcn_snn else ("Model not loaded", np.zeros(4))

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Pure TCN**")
                st.markdown(f"Prediction: **{pred_tcn}**")
                for l, p in zip(labels, prob_tcn):
                    st.write(f"{l}: {p*100:.1f}%")
                    st.progress(float(p))

            with col_b:
                st.markdown("**Hybrid TCN-SNN**")
                st.markdown(f"Prediction: **{pred_snn}**")
                for l, p in zip(labels, prob_snn):
                    st.write(f"{l}: {p*100:.1f}%")
                    st.progress(float(p))

# ---------------------------------------------------------
# MODEL EXPLANATION (INPUT + GRAD-CAM COMPARISON)
# ---------------------------------------------------------
st.divider()

with st.expander("Model explanation (Input & attention comparison)", expanded=False):

    if mel is not None and pure_tcn is not None and tcn_snn is not None:

        LABELS = ["COPD", "Healthy", "Pneumonia", "Other"]

        # Predictions (already computed earlier, reused safely)
        cls_idx_tcn = LABELS.index(pred_tcn)
        cls_idx_snn = LABELS.index(pred_snn)

        x_tensor = torch.tensor(mel).unsqueeze(0).float()

        cam_tcn = GradCAM(pure_tcn)
        cam_snn = GradCAM(tcn_snn)

        heatmap_tcn = cam_tcn.generate(x_tensor, cls_idx_tcn)
        heatmap_snn = cam_snn.generate(x_tensor, cls_idx_snn)

        col1, col2, col3 = st.columns([1.2, 1, 1])

        # --- Input Mel ---
        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title("Input: Mel-spectrogram", fontsize=10)
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency")
            ax.tick_params(labelsize=8)
            st.pyplot(fig, clear_figure=True)

        # --- Pure TCN CAM ---
        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(
                heatmap_tcn,
                aspect="auto",
                origin="lower",
                cmap="inferno",
                vmin=0,
                vmax=1,
            )
            ax.set_title("Pure TCN — Grad-CAM", fontsize=10)
            ax.set_xlabel("Time")
            ax.set_ylabel("Features")
            ax.tick_params(labelsize=8)
            st.pyplot(fig, clear_figure=True)

        # --- Hybrid TCN-SNN CAM ---
        with col3:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(
                heatmap_snn,
                aspect="auto",
                origin="lower",
                cmap="inferno",
                vmin=0,
                vmax=1,
            )
            ax.set_title("Hybrid TCN-SNN — Grad-CAM", fontsize=10)
            ax.set_xlabel("Time")
            ax.set_ylabel("Features")
            ax.tick_params(labelsize=8)
            st.pyplot(fig, clear_figure=True)

    else:
        st.info("Run analysis first to view model explanations.")


# ---------------------------------------------------------
# RESET BUTTON
# ---------------------------------------------------------
st.divider()
if st.button("Analyze another recording"):
    # Clear session state and rerun
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.experimental_rerun()

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; margin-top:40px; color:#9ca3af; font-size:12px;">
        PulmoScope © 2025<br>
        Academic prototype — not a medical device
    </div>
    """,
    unsafe_allow_html=True,
)
