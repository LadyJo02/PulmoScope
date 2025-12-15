<p align="center">
  <img src="assets/slides/1_title.png" width="90%">
  <img src="assets/slides/2_problem.png" width="90%">
  <img src="assets/slides/3_gaps.png" width="90%">
  <img src="assets/slides/4_objectives.png" width="90%">
  <img src="assets/slides/5_methodology.png" width="90%">
  <img src="assets/slides/6_stage1.png" width="90%">
  <img src="assets/slides/7_stage2.png" width="90%">
  <img src="assets/slides/8_stage3.png" width="90%">
  <img src="assets/slides/9_stage4.png" width="90%">
  <img src="assets/slides/10_stage5.png" width="90%">
  <img src="assets/slides/11_eda.png" width="90%">
  <img src="assets/slides/12_data.png" width="90%">
  <img src="assets/slides/13_healthy.png" width="90%">
  <img src="assets/slides/14_copd.png" width="90%">
  <img src="assets/slides/15_pneumonia.png" width="90%">
  <img src="assets/slides/16_centroid.png" width="90%">
  <img src="assets/slides/17_bandwidth.png" width="90%">
  <img src="assets/slides/18_rolloff.png" width="90%">
  <img src="assets/slides/19_flux.png" width="90%">
  <img src="assets/slides/20_results.png" width="90%">
  <img src="assets/slides/21_metrics.png" width="90%">
  <img src="assets/slides/22_test.png" width="90%">
  <img src="assets/slides/23_gradcam.png" width="90%">
  <img src="assets/slides/24_conclusion.png" width="90%">
  <img src="assets/slides/25_last.png" width="90%">
</p>

---

# PulmoScope — AI-Assisted Lung Sound Analysis  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)
![Deep Learning](https://img.shields.io/badge/AI-TCN%20%7C%20TCN--SNN-green)
![License](https://img.shields.io/badge/License-Academic-lightgrey)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

### Live Demo: https://pulmoscope.streamlit.app/
### Full Manuscript, Slides, and Supporting Notebooks

The complete research materials for PulmoScope are provided below for reference, transparency, and reproducibility.

**Slide Presentation**
- [PulmoScope Project Slides](https://tinyurl.com/canvascope)

**Manuscript**
- [DL-FINAL-PROJECT-PULMOSCOPE.pdf](assets/manuscript/DL-FINAL-PROJECT-PULMOSCOPE.pdf)

**Exploratory and Model Analysis Notebooks**
- [EDA_Pulmoscope.ipynb](assets/notebooks/EDA_Pulmoscope.ipynb) — exploratory data analysis and feature inspection  
- [PulmoScope.ipynb](assets/notebooks/PulmoScope.ipynb) — model development and experimentation  

---

## 1. Project Overview

PulmoScope is a **deep-learning–based assistive system** for analyzing lung auscultation sounds and classifying respiratory conditions.  
The system is designed to support clinical screening by leveraging **temporal deep learning models** for disease-level lung sound classification.It compares two architectures:
- **Pure Temporal Convolutional Network (TCN)**
- **Hybrid Temporal Convolutional Network with Spiking Neural Network (TCN–SNN)**

**Disclaimer:** PulmoScope is not a medical device. It is intended solely for academic research and demonstration.

---

## 2. System Pipeline

![Pipeline Overview](assets/figures/pipeline_overview.png)

**Processing stages:**
1. Lung sound acquisition  
2. Signal preprocessing  
3. Mel-spectrogram feature extraction  
4. Temporal model inference  
5. Prediction and interpretability  

---

## 3. Dataset and Exploratory Analysis

PulmoScope is evaluated using the **ICBHI 2017 Respiratory Sound Database**, containing labeled lung sound recordings across multiple respiratory conditions.

---

## 4. Model Architecture

### Pure TCN
- Kernel sizes: 3, 5, 7
- Dilated convolutions
- Residual blocks
- Attention module

### Hybrid TCN–SNN
- Identical TCN backbone
- Parametric LIF spiking neuron module
- Sparse temporal activation
- Attention classifier head

### Shared Classifier
- Dense layers: 192 → 128 → 64
- GELU activation
- Dropout
- Softmax output layer

---

## 5. Repository Structure

```
PulmoScope/
├── app.py                            # Streamlit web application entry point
├── assets/                             
│   ├── banner.png                    # Application header/banner image
│   └── figures/                      # Figures used in manuscript and README
│   ├── manuscript/                   # Complete Manuscript
│   ├── notebooks/                    # Exploratory and Model Analysis Notebooks
│   ├── sample_audio/                 # Sample audio for streamlit
│   ├── slides/                       # PPT slide PNGs for README
├── models/                           
│   ├── pure_tcn_config.json          # Pure TCN architecture configuration
│   ├── tcn_snn_config.json           # Hybrid TCN–SNN architecture configuration
│   ├── pure_tcn_weights.pth          # Trained Pure TCN model weights
│   └── tcn_snn_weights.pth           # Trained Hybrid TCN–SNN model weights
├── utils/                            
│   ├── preprocess.py                 # Audio loading and mel-spectrogram extraction
│   ├── inference.py                  # Model loading and prediction logic
│   ├── architectures.py              # TCN and TCN–SNN model definitions
│   ├── gradcam.py                    # Grad-CAM attention visualization
│   └── audio_utils.py                # Audio helper and signal utilities
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
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

## 7. Demo Workflow (L.U.N.G Framework)

PulmoScope follows a simple user interaction flow:

- **L** – Load lung sound  
- **U** – Understand sound patterns using AI  
- **N** – Notify likely condition  
- **G** – Guide clinical decision support  

Live demo available at:  
https://pulmoscope.streamlit.app/

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

This project is open source and available under the [MIT License](LICENSE).
