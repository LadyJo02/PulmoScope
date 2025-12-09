# 🫁 PulmoScope

**PulmoScope** is an academic deep-learning web application for **AI-assisted analysis of lung sound recordings**.  
It enables automated classification of respiratory conditions using **Temporal Convolutional Networks (TCN)** and a **hybrid TCN–Spiking Neural Network (TCN-SNN)** model.

> ⚠️ **Disclaimer**  
> PulmoScope is an academic prototype only and **not a medical device**.  
> It is intended for research and educational purposes.

---

## 🚀 Features

- Upload lung sound recordings (`.wav`)  
- Optional **real-time audio recording** via microphone  
- Compare **two deep-learning models side-by-side**  
- Mel-spectrogram visualization  
- Multi-class prediction: Normal, COPD, Pneumonia, Other Respiratory Conditions  
- Streamlit-based UI (deployable on Streamlit Cloud)  

---

## 🧠 Models

Two trained models are supported:

| Model | Description |
|-----|------------|
| **Pure TCN** | Multi-scale Temporal Convolutional Network with attention |
| **Hybrid TCN–SNN** | TCN backbone combined with Spiking Neural Network dynamics |

Both models were trained under a **strict apples-to-apples experimental setup**, sharing:
- Same preprocessing pipeline  
- Same classifier head  
- Same hyperparameter search strategy  

---

## 📁 Project Structure

```text
PulmoScope/
├── app.py                     # Streamlit web app
├── assets/
│   └── banner.png              # UI header banner
├── models/
│   ├── pure_tcn_weights.pth
│   └── tcn_snn_weights.pth
├── utils/
│   ├── preprocess.py           # Audio preprocessing
│   ├── inference.py            # Model loading and inference
│   ├── architectures.py        # Model architectures
│   └── gradcam.py              # Grad-CAM explainer (optional)
├── requirements.txt
└── README.md
