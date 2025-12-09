# utils/gradcam.py
import torch
import numpy as np

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        layer = model.tcn3.bn
        layer.register_forward_hook(self._save_activation)
        layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, class_idx].backward()

        # Channel-wise importance
        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (self.activations * weights).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam
