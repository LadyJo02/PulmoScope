import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocess import load_audio, create_mel
from utils.inference import load_model, predict
from PIL import Image

st.set_page_config(
    page_title="PulmoScope",
    page_icon="🩺",
    layout="centered"
)

# Header Image
st.image("assets/banner.png", use_container_width=True)

st.markdown(
    """
    <h1 style='text-align:center; color:#123C4C;'>PulmoScope</h1>
    <p style='text-align:center; font-size:18px;'>
        AI-assisted analysis of lung sound recordings.
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------
# MODEL SELECTION
# -------------------------
st.subheader("Choose a Model")

model_choice = st.selectbox(
    "Select a classification model:",
    (
        "Pure TCN Model",
        "Hybrid TCN-SNN Model"
    )
)

MODEL_PATHS = {
    "Pure TCN Model": "models/pure_tcn_weights.pth",
    "Hybrid TCN-SNN Model": "models/tcn_snn_weights.pth"
}

model_path = MODEL_PATHS[model_choice]
model = load_model(model_path)


# -------------------------
# FILE UPLOADER
# -------------------------
uploaded_file = st.file_uploader("Upload a lung sound WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    with st.spinner("Processing audio..."):
        audio = load_audio(uploaded_file)
        mel = create_mel(audio)

    st.subheader("Mel-Spectrogram")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(mel, origin="lower", cmap="viridis", aspect="auto")
    ax.set_title("Mel-Spectrogram")
    st.pyplot(fig)

    with st.spinner("Analyzing sound..."):
        prediction, probabilities = predict(model, mel)

    st.success(f"Prediction: **{prediction}**")

    st.write("### Confidence Levels")
    st.write(f"- Normal: `{probabilities[0]*100:.2f}%`")
    st.write(f"- COPD: `{probabilities[1]*100:.2f}%`")
    st.write(f"- Pneumonia: `{probabilities[2]*100:.2f}%`")

st.markdown(
    """
    <hr>
    <p style='text-align:center; color:gray; font-size:12px;'>
        PulmoScope © 2025 <br>
        Academic prototype only — Not a medical device.
    </p>
    """,
    unsafe_allow_html=True
)
