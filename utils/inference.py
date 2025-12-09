import torch
import numpy as np

LABELS = ["Normal", "COPD", "Pneumonia"]

# TEMPORARY MOCK LOADER
def load_model(path="model/pulmonary_cnn.pth"):
    try:
        model = torch.load(path, map_location=torch.device("cpu"))
        model.eval()
        return model
    except Exception:
        print("⚠ Model file not found. Using MOCK model for testing.")
        return None


# TEMPORARY MOCK PREDICTOR
def predict(model, mel):
    # If no model → return fake prediction
    if model is None:
        print("⚠ Using mock prediction.")
        fake_probs = np.array([0.33, 0.33, 0.34])
        pred = LABELS[np.argmax(fake_probs)]
        return pred, fake_probs

    # REAL INFERENCE (once model is available)
    x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].numpy()

    pred = LABELS[np.argmax(prob)]
    return pred, prob
