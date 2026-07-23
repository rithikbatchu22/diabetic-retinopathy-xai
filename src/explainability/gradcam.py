from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_last_conv_layer(model: nn.Module) -> Tuple[str, nn.Module]:
    """
    Find the last nn.Conv2d layer in a model.

    Why:
    Grad-CAM needs a convolutional feature map layer (spatial map).
    Using the last conv layer usually produces the most class-specific heatmap.
    """
    last_name, last_layer = None, None
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            last_name, last_layer = name, layer
    if last_layer is None:
        raise ValueError("No nn.Conv2d layer found in the model.")
    return last_name, last_layer


@dataclass
class GradCAMOutput:
    class_idx: int
    heatmap: np.ndarray          # (H, W) float32 in [0,1]
    logits: np.ndarray           # (num_classes,)
    overlay_rgb: np.ndarray      # (H, W, 3) uint8
    heatmap_rgb: np.ndarray      # (H, W, 3) uint8


class GradCAM:
    """
    Minimal Grad-CAM implementation.

    Usage:
      cam = GradCAM(model)
      out = cam.generate(input_tensor, original_image_rgb_uint8)
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            _, target_layer = find_last_conv_layer(model)
        self.target_layer = target_layer

        self._activations = None
        self._gradients = None
        self._hooks = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self._activations = output

        def backward_hook(_, grad_input, grad_output):
            # grad_output[0] has shape [B, C, h, w]
            self._gradients = grad_output[0]

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        # full_backward_hook is preferred in newer PyTorch
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _make_heatmap(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Grad-CAM heatmap (tensor) in [0,1] and upsample to input size.
        Returns heatmap of shape [B, H, W].
        """
        if self._activations is None or self._gradients is None:
            raise RuntimeError("Missing activations/gradients. Did you call backward()?")

        # activations: [B, C, h, w], gradients: [B, C, h, w]
        grads = self._gradients
        acts = self._activations

        # weights: global-average-pool gradients over spatial dims -> [B, C, 1, 1]
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # weighted sum over channels -> [B, h, w]
        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)

        # normalize per-sample to [0,1]
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)

        # upsample to input size
        cam = cam.unsqueeze(1)  # [B,1,h,w]
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)    # [B,H,W]
        return cam

    @torch.enable_grad()
    def generate(
        self,
        x: torch.Tensor,
        original_rgb_uint8: np.ndarray,
        class_idx: Optional[int] = None,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> GradCAMOutput:
        """
        x:
          torch.Tensor of shape [1, 3, H, W] already normalized like during inference.
        original_rgb_uint8:
          numpy RGB image (H, W, 3) uint8 for visualization overlay.

        Returns:
          GradCAMOutput with heatmap + overlays.
        """
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError("x must have shape [1, 3, H, W]")

        device = next(self.model.parameters()).device
        x = x.to(device)

        # Forward
        logits = self.model(x)  # [1, num_classes]

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        # Backward on the chosen class score
        score = logits[:, class_idx].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        # Heatmap tensor -> numpy
        heatmap_t = self._make_heatmap(x)[0]  # [H,W]
        heatmap = heatmap_t.detach().cpu().numpy().astype(np.float32)

        # Create colored heatmap + overlay
        overlay_rgb, heatmap_rgb = overlay_heatmap_on_rgb(
            original_rgb_uint8, heatmap, alpha=alpha, colormap=colormap
        )

        return GradCAMOutput(
            class_idx=class_idx,
            heatmap=heatmap,
            logits=logits[0].detach().cpu().numpy(),
            overlay_rgb=overlay_rgb,
            heatmap_rgb=heatmap_rgb,
        )


def overlay_heatmap_on_rgb(
    img_rgb_uint8: np.ndarray,
    heatmap_01: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    img_rgb_uint8: (H,W,3) uint8
    heatmap_01: (H,W) float in [0,1]

    Returns:
      overlay_rgb_uint8, heatmap_rgb_uint8
    """
    if img_rgb_uint8.dtype != np.uint8:
        raise ValueError("img_rgb_uint8 must be uint8 RGB image")
    if heatmap_01.ndim != 2:
        raise ValueError("heatmap_01 must have shape (H,W)")

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_01, 0, 1))
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_rgb_uint8, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return overlay, heatmap_rgb