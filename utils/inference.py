import torch
import numpy as np
from utils.architectures import Model_Pure_TCN, Model_TCN_SNN

LABELS = ["Normal","COPD","Pneumonia","Other"]

def load_model(path, model_type):
    model = (Model_Pure_TCN() if model_type=="tcn"
            else Model_TCN_SNN())
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model

def predict(model, mel):
    x = torch.tensor(mel).unsqueeze(0).float()
    with torch.no_grad():
        logits = model(x)
        p = torch.softmax(logits,1)[0].numpy()
    return LABELS[p.argmax()], p
