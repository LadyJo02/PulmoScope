import torch
import json
import numpy as np
from utils.architectures import Model_Pure_TCN, Model_TCN_SNN

LABELS = ["Normal", "COPD", "Pneumonia", "Other"]


def load_model(weight_path: str, model_type: str):
    """
    Load model using the saved deployment config to ensure
    EXACT architecture match.
    """
    config_path = weight_path.replace("_weights.pth", "_config.json")

    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)

        hp = cfg["model_hyperparameters"]

        if model_type == "tcn":
            model = Model_Pure_TCN(
                input_channels=hp["input_channels"],
                hidden_dim=hp["hidden_dim"],
                dropout=hp["dropout"],
                n_classes=cfg["n_classes"],
            )
        else:
            model = Model_TCN_SNN(
                input_channels=hp["input_channels"],
                hidden_dim=hp["hidden_dim"],
                dropout=hp["dropout"],
                n_classes=cfg["n_classes"],
            )

        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return None


def predict(model, features: np.ndarray):
    if model is None:
        probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        return "Model not loaded", probs

    x = torch.tensor(features).unsqueeze(0).float()  # (1, 264, T)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    return LABELS[int(np.argmax(probs))], probs
