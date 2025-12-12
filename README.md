# 🫁 PulmoScope — AI-Assisted Lung Sound Analysis  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)
![Deep Learning](https://img.shields.io/badge/AI-TCN%20%7C%20TCN--SNN-green)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

PulmoScope is a **deep-learning powered assistive tool** that analyzes **lung auscultation sounds** to classify common respiratory conditions such as **COPD**, **Pneumonia**, **Healthy**, and **Other respiratory abnormalities**.

Developed as part of a university research project, PulmoScope implements **advanced temporal AI models** including a **Pure Temporal Convolutional Network (TCN)** and a **Hybrid TCN–Spiking Neural Network (TCN-SNN)**.

> ⚠️ **Disclaimer:**  
> PulmoScope is **not a medical device**.  
> It is intended for academic research and demonstration only.

---

## 📌 1. Project Overview

PulmoScope enables:

- Uploading of `.wav` lung sound recordings  
- Optional **in-browser live recording**  
- Automatic preprocessing, filtering, and spectrogram generation  
- Side-by-side comparison of **TCN** and **TCN-SNN** predictions  
- Grad-CAM–based interpretability heatmaps  
- A clean, interactive Streamlit UI

The models were developed using a rigorous **3-phase experimental framework** including architecture search, hyperparameter tuning, and held-out testing.

---

## 🚀 2. Features

### 🔊 Audio Input
- Upload `.wav` recordings  
- Record directly using microphone input  

### 🧠 AI Models
- **Pure TCN:** Multi-scale temporal convolution + attention  
- **Hybrid TCN–SNN:** Temporal convolution + spiking neuron dynamics  

### 📊 Visualizations
- Mel-spectrograms  
- Grad-CAM heatmaps (optional)  
- Model probability bars  

### 🏥 Diagnostic Categories
| Label | Description |
|-------|-------------|
| **Healthy** | No audible abnormal sounds |
| **COPD** | Continuous wheezes, airflow obstruction |
| **Pneumonia** | Crackles, fluid-related abnormalities |
| **Other** | Asthma, URTI/LRTI, Bronchiectasis |

---

## 🧠 3. Model Overview

### **🔹 Pure TCN Model**
- Multi-scale convolution kernels (3, 5, 7)  
- Dilated layers for long-range temporal context  
- Residual connections  
- Standard attention mechanism  

### **🔹 Hybrid TCN–SNN Model**
- Identical TCN backbone as the Pure TCN  
- Final feature stage converted into spike-based representations  
- Parametric LIF neuron for noise-resilient decision-making  
- Attention-based classifier head  

### Shared Classifier Head
Both architectures share:
- 192 → 128 → 64 Dense layers  
- GELU activations  
- Dropout regularization  
- Final softmax for 4-class prediction  

---

## 📁 4. Repository Structure

```text
PulmoScope/
├── app.py                     # Streamlit app entry point
├── assets/
│   ├── banner.png             # App banner
│   └── icons/                 # Optional UI icons
├── models/
│   ├── pure_tcn_weights.pth
│   └── tcn_snn_weights.pth
├── utils/
│   ├── preprocess.py          # Filtering, segmentation, feature extraction
│   ├── inference.py           # Model loading + inference pipeline
│   ├── architectures.py       # TCN, TCN-SNN definitions
│   ├── gradcam.py             # Grad-CAM explainability
│   └── audio_utils.py         # Helper functions
├── requirements.txt
└── README.md
