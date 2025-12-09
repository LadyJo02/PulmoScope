import torch
import numpy as np
from utils.architectures import Model_Pure_TCN, Model_TCN_SNN

LABELS = ["Normal", "COPD", "Pneumonia", "Other"]


def load_model(path: str, model_type: str):
    """
    Rebuild the architecture and load the saved state_dict.
    model_type: "tcn" or "snn"
    """
    if model_type == "tcn":
        model = Model_Pure_TCN()
    else:
        model = Model_TCN_SNN()

    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model, features: np.ndarray):
    """
    features: numpy array (C=264, T)
    """
    x = torch.tensor(features).unsqueeze(0).float()  # (1, C, T)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return LABELS[pred_idx], probs
