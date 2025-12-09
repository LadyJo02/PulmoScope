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

# Load Header Image
st.image("assets/banner.png", use_container_width=True)

st.markdown(
    """
    <h1 style='text-align:center; color:#4A90E2;'>PulmoScope</h1>
    <p style='text-align:center; font-size:18px;'>
        Upload a lung sound recording and let the AI analyze it.
    </p>
    """,
    unsafe_allow_html=True
)

# Load Model
model = load_model("model/pulmonary_cnn.pth")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

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
