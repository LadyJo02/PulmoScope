# utils/gradcam.py
import torch
import numpy as np

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        self.target_layer = model.tcn3.bn

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx):
        self.model.zero_grad()

        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward()

        # Grad-CAM weights
        weights = self.gradients.mean(dim=2, keepdim=True)

        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        # Expand CAM to 2D heatmap for visualization
        cam = np.repeat(cam[np.newaxis, :], x.shape[1], axis=0)
        return cam
