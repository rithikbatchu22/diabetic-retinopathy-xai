from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union
import contextlib

import numpy as np
import torch

from src.data.preprocessing import PreprocessConfig, preprocess_image
from src.data.augmentation import AugmentConfig, get_val_transforms
from src.explainability.gradcam import GradCAM
from src.models.model import create_dr_model


CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


@dataclass
class XAIResult:
    pred_class: int
    pred_label: str
    confidence: float
    probs: np.ndarray
    overlay_rgb: np.ndarray
    heatmap_rgb: np.ndarray


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model_name: str = "efficientnet_b0",
    num_classes: int = 5,
    device: str | None = None,
):
    """
    IMPORTANT:
    Do NOT use torch.inference_mode here. Grad-CAM needs backward later.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = create_dr_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, device


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def predict_with_gradcam(
    model,
    device: str,
    image_path: Union[str, Path],
    checkpoint_image_size: int = 384,
) -> XAIResult:
    """
    Robust Grad-CAM prediction.

    Key fix:
    - We explicitly DISABLE torch.inference_mode() inside this function,
      so even if the caller accidentally runs under inference_mode,
      the tensors used for Grad-CAM are normal autograd tensors.
    """

    # Some PyTorch versions support torch.inference_mode(False). If not, do nothing.
    try:
        infer_off_ctx = torch.inference_mode(False)
    except TypeError:
        infer_off_ctx = contextlib.nullcontext()

    with infer_off_ctx:
        # 1) Preprocess (uint8 RGB)
        pre_cfg = PreprocessConfig(
            image_size=checkpoint_image_size,
            crop_black=True,
            enhance_contrast=False,
        )
        img_rgb = preprocess_image(image_path, pre_cfg)

        # 2) Model input tensor (created OUTSIDE inference_mode)
        tfm = get_val_transforms(AugmentConfig(image_size=checkpoint_image_size))
        x = tfm(image=img_rgb)["image"].unsqueeze(0).to(device)

        # 3) Grad-CAM forward+backward
        cam = GradCAM(model)
        gc = cam.generate(x, img_rgb)  # chooses argmax class by default
        cam.remove_hooks()

        probs = _softmax_np(gc.logits)
        pred_class = int(gc.class_idx)
        confidence = float(probs[pred_class])

        return XAIResult(
            pred_class=pred_class,
            pred_label=CLASS_NAMES[pred_class],
            confidence=confidence,
            probs=probs,
            overlay_rgb=gc.overlay_rgb,
            heatmap_rgb=gc.heatmap_rgb,
        )