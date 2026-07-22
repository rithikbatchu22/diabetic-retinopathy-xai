from __future__ import annotations

import torch
import torch.nn as nn
import timm


def create_dr_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 5,
    pretrained: bool = True,
) -> nn.Module:
    """
    Create a DR severity classification model using timm backbones.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=3,
    )
    return model


@torch.inference_mode()
def predict_proba(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return softmax probabilities."""
    logits = model(x)
    return torch.softmax(logits, dim=1)