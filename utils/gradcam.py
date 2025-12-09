import torch
import numpy as np


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Target: last TCN block
        target_layer = model.tcn3.bn
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output  # (B, C, T)

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # (B, C, T)

    def generate(self, x, class_idx):
        """
        x: Tensor of shape (1, C, T)
        """
        self.model.zero_grad()

        logits = self.model(x)
        logits[0, class_idx].backward()

        # Global average pooling over time
        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, C, 1)

        cam = (self.activations * weights).sum(dim=1)       # (1, T)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = np.maximum(cam, 0)
        cam /= cam.max() + 1e-8

        # Expand over feature axis for visualization
        cam = np.tile(cam, (x.shape[1], 1))  # (C, T)

        return cam
