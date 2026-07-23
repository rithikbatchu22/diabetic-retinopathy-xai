from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

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


@torch.inference_mode()
def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model_name: str = "efficientnet_b0",
    num_classes: int = 5,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = create_dr_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, device


def predict_with_gradcam(
    model,
    device: str,
    image_path: Union[str, Path],
    checkpoint_image_size: int = 384,
) -> XAIResult:
    # 1) preprocess for visualization + model input
    pre_cfg = PreprocessConfig(image_size=checkpoint_image_size, crop_black=True, enhance_contrast=False)
    img_rgb = preprocess_image(image_path, pre_cfg)  # uint8 RGB

    # 2) model input transform (normalize + tensor)
    tfm = get_val_transforms(AugmentConfig(image_size=checkpoint_image_size))
    x = tfm(image=img_rgb)["image"].unsqueeze(0).to(device)

    # 3) prediction
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class])

    # 4) Grad-CAM (needs gradients)
    cam = GradCAM(model)
    out = cam.generate(x, img_rgb, class_idx=pred_class)

    return XAIResult(
        pred_class=pred_class,
        pred_label=CLASS_NAMES[pred_class],
        confidence=confidence,
        probs=probs,
        overlay_rgb=out.overlay_rgb,
        heatmap_rgb=out.heatmap_rgb,
    )