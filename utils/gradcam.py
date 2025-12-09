import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grad = None
        self.act = None
        self.hooks = []

        layer = model.tcn3.bn
        self.hooks.append(layer.register_forward_hook(self._save_act))
        self.hooks.append(layer.register_backward_hook(self._save_grad))

    def _save_act(self, m, i, o): self.act = o
    def _save_grad(self, m, gi, go): self.grad = go[0]

    def generate(self, x, cls):
        self.model.zero_grad()
        out = self.model(x)
        out[0, cls].backward()

        w = self.grad.mean(dim=2, keepdim=True)
        cam = (self.act*w).mean(1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam,0)
        cam /= cam.max()+1e-8
        cam = cv2.resize(cam,(x.shape[-1],1))
        return np.tile(cam,(x.shape[2],1))
