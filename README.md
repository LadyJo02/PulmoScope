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

```
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
```

---

## 6. Installation

### Clone the Repository
```
git clone https://github.com/LadyJo02/PulmoScope.git
cd PulmoScope
```

### Install Dependencies
```
pip install -r requirements.txt
```

### Run the Application
```
streamlit run app.py
```

---

## 7. Usage

1. Launch the Streamlit app.
2. Upload a `.wav` file or record new audio.
3. PulmoScope performs preprocessing:
   - Resampling  
   - Filtering  
   - Segmentation  
   - Mel-spectrogram extraction  
4. Both TCN and TCN-SNN generate predictions.
5. View probability outputs and optional Grad-CAM heatmaps.

---

## 8. Future Improvements

- Digital stethoscope integration  
- Mobile deployment  
- Noise-robust denoising models  
- Larger clinical dataset expansion  

---

## 9. Authors

PulmoScope was developed by:

- Genheylou Felisilda  
- Nicole Menorias  
- Kobe Marco Olaguir  
- Joanna Reyda Santos

---

## 10. License

This project is intended solely for academic research and educational purposes. It is not approved for clinical use.
