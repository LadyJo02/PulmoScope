import torch
import numpy as np

LABELS = ["Normal", "COPD", "Pneumonia"]

def load_model(path):
    """
    Loads a PyTorch model if available.
    Returns None if missing (mock mode).
    """
    try:
        model = torch.load(path, map_location=torch.device("cpu"))
        model.eval()
        print(f"Loaded model: {path}")
        return model
    except Exception as e:
        print(f"⚠ Model not found ({path}). Using MOCK model.")
        return None


def predict(model, mel):
    """
    Runs prediction.
    Uses mock prediction if model is None.
    """

    # MOCK PREDICTION
    if model is None:
        print("⚠ Using mock prediction.")
        fake_probs = np.array([0.34, 0.33, 0.33])  # Normal, COPD, Pneumonia
        pred = LABELS[np.argmax(fake_probs)]
        return pred, fake_probs

    # REAL PREDICTION
    x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()
    
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].numpy()

    pred = LABELS[np.argmax(prob)]
    return pred, prob
